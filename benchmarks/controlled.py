"""Six-method controlled experiments with frozen method-specific settings.

``python -m benchmarks.controlled --output /path/to/run --workers 8`` runs the
full 10-base confirmation grid. ``--limit 1 --smoke`` performs a technical
test only, using one training epoch and two probes. Such a test cannot be
reported as a research result. All outputs are written outside the source tree
when an external --output directory is supplied.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from geopr_vae import reference,scoring
from geopr_vae.pipeline import CONFIGS
from geopr_vae.training import train_residual_head
from .controlled_data import generate_dataset,construct_input_arrays
from .metrics import readout,evaluate_arrays
from .patch_runner import fit_model,predict_fields

DEFAULT_CONFIG=Path(__file__).resolve().parents[1]/'configs/paper_experiments.json'

def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def train_reference(a,method,seed,epochs):
    native='RAMP_VAE' if method=='GeoPR-VAE' else method
    model=reference.make_model(native,CONFIGS[method],int(seed))
    x=torch.as_tensor(a['train_x'],dtype=torch.float32); c=torch.as_tensor(a['train_c'],dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed+941)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True)
        terms=model.objective(x,c,generator=generator,epoch_fraction=epoch/epochs)
        loss=terms['total_loss']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
    model.eval();return model

def predict_reference(model,a):
    p={}
    for role in ('train','cal'):
        p[role+'_reference']=model.predict(a[role+'_x'],a[role+'_c'],ids=a[role+'_ids'],mc_samples=16)['reference']
    fields=np.concatenate((a['val_x'][None],a['val_x'][None]+a['deltas']))
    n=len(fields)
    pred=model.predict(fields.reshape(-1,12),np.tile(a['val_c'],(n,1)),ids=np.tile(a['val_ids'],n),mc_samples=16)
    p['reference']=pred['reference'].reshape(fields.shape)
    return p

def field_standardize(r):
    location=np.median(r,axis=-2,keepdims=True)
    scale=np.maximum(1.4826*np.median(np.abs(r-location),axis=-2,keepdims=True),.1)
    return (r-location)/scale

def score_fields(carrier,cal_carrier,a,cfg):
    if cfg['geometry']!='gaussian':raise ValueError('This frozen experiment uses Gaussian scoring')
    norm=cfg['norm']
    if norm=='field_mad':
        carrier=field_standardize(carrier);cal_carrier=np.array(cal_carrier,copy=True)
        for ix in scoring.split_calibration(a['cal_ids']):cal_carrier[ix]=field_standardize(cal_carrier[ix])
        norm='cal_mad'
    out=scoring.score_fields(carrier,cal_carrier,a['val_xy'],a['cal_xy'],a['cal_ids'],norm,cfg['radius_m'],pool='threshold',power_correction=False)
    transform=scoring.fit_affine_map(out['cal_score'])
    out['score']=scoring.affine_map(out['score'],transform)
    out['cal_score']=scoring.affine_map(out['cal_score'],transform)
    out['threshold']=float(scoring.affine_map(out['threshold'],transform))
    return out

def run_task(payload):
    task,cfg,output,smoke,save_predictions=payload
    torch.set_num_threads(1)
    method=task['method'];spec=cfg['methods'][method];fold=task['fold']
    d=generate_dataset(task['family'],task['base_seed'],task['noise'])
    a,meta=construct_input_arrays(d,fold,task['buffer_m'])
    if smoke:
        a['deltas']=a['deltas'][:2];a['receivers']=a['receivers'][:2]
        meta['cases']=meta['cases'][:2]
    offset=cfg['fold_seed_offset']*fold
    if method in CONFIGS:
        model=train_reference(a,method,cfg['reference_seed']+offset,1 if smoke else cfg['reference_epochs'])
        p=predict_reference(model,a)
    else:
        model,low,span=fit_model(a,method,cfg['patch_seed']+offset,1 if smoke else cfg['patch_epochs'])
        p=predict_fields(model,a,low,span)
    fields=np.concatenate((a['val_x'][None],a['val_x'][None]+a['deltas']))
    carrier=fields-p['reference'];cc=a['cal_x']-p['cal_reference']
    if method=='GeoPR-VAE':
        head,_=train_residual_head(a['train_x'],a['train_c'],a['train_xy'],p['train_reference'],
            cfg['recovery_seed']+offset,spec['head']['config'],1 if smoke else cfg['recovery_epochs'])
        carrier=head.predict(carrier,a['val_c'],a['val_xy'])['correction']
        for ix in scoring.split_calibration(a['cal_ids']):
            cc[ix]=head.predict(cc[ix],a['cal_c'][ix],a['cal_xy'][ix])['correction']
        denoised=p['reference']+carrier
    else:denoised=readout(fields,p['reference'],spec['readout'])
    scored=score_fields(carrier,cc,a,spec['score'])
    summary,metrics=evaluate_arrays(a,p['reference'],denoised,scored)
    out=Path(output)/task['id'];out.mkdir(parents=True,exist_ok=True)
    record=dict(task=task,technical_smoke=smoke,summary=summary,settings=spec,
                probe_cases=meta['cases'],reference_epochs=(1 if smoke else cfg['reference_epochs']) if method in CONFIGS else None,
                patch_epochs=(1 if smoke else cfg['patch_epochs']) if method not in CONFIGS else None,
                recovery_epochs=(1 if smoke else cfg['recovery_epochs']) if method=='GeoPR-VAE' else None)
    (out/'metrics.json').write_text(json.dumps(jsonable(record),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'case_metrics.npz',**metrics)
    if save_predictions:np.savez_compressed(out/'predictions.npz',reference=p['reference'],denoised=denoised,cal_reference=p['cal_reference'])
    return dict(task=task,summary=summary,technical_smoke=smoke)

def tasks(cfg,phase,methods):
    for seed in cfg[phase+'_seeds']:
        for family in cfg['families']:
            for noise in cfg['noises']:
                for fold in cfg['folds']:
                    for buffer in cfg['buffers_m']:
                        for method in methods:
                            yield dict(id=f'{method}_s{seed}_{family}_{noise}_f{fold}_b{buffer}',method=method,
                                       base_seed=seed,family=family,noise=noise,fold=fold,buffer_m=buffer)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=DEFAULT_CONFIG);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--methods',nargs='+');ap.add_argument('--phase',choices=['development','confirmation'],default='confirmation')
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--limit',type=int);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--save-predictions',action='store_true')
    args=ap.parse_args();cfg=json.loads(args.config.read_text());methods=args.methods or list(cfg['methods'])
    if not 1<=args.workers<=8:ap.error('workers must be between 1 and 8')
    if not set(methods)<=set(cfg['methods']):ap.error('Unknown methods')
    jobs=list(tasks(cfg,args.phase,methods))
    if args.limit is not None:jobs=jobs[:args.limit]
    if args.smoke and args.limit is None:ap.error('--smoke requires an explicit --limit')
    args.output.mkdir(parents=True,exist_ok=True)
    payloads=[(t,cfg,str(args.output),args.smoke,args.save_predictions) for t in jobs]
    results=[]
    if args.workers==1:
        for item in payloads:results.append(run_task(item));print('completed',len(results),'/',len(jobs),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(run_task,p) for p in payloads]):
                results.append(f.result());print('completed',len(results),'/',len(jobs),flush=True)
    (args.output/'summary.json').write_text(json.dumps(jsonable(dict(phase=args.phase,technical_smoke=args.smoke,rows=results)),indent=2)+'\n')

if __name__=='__main__':main()
