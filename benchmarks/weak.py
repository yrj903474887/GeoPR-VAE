"""Prepare and evaluate the fixed regional paired-injection experiment.

prepare: input is a patch_runner NPZ; output additionally contains 144 deltas,
positive_mask, negative_mask and case_ids. Run patch_runner with this augmented
input. evaluate: combine that input and predictions.npz to evaluate the original
and perturbed native scores. The same model is used for all 145 fields.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .injections import TARGETS,build_injections
from .metrics import binary_metrics,recall_budget

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fold',type=int,required=True);p.add_argument('--buffer-km',type=int,choices=[4,8],required=True)
    e=sub.add_parser('evaluate');e.add_argument('--input',type=Path,required=True);e.add_argument('--predictions',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('core-predict',help='Paired inference using a trained AE/VAE/GeoPR-VAE package bundle')
    c.add_argument('--input',type=Path,required=True,help='Prepared injection input')
    c.add_argument('--query',type=Path,required=True,help='Query observations in geopr_vae input schema')
    c.add_argument('--model',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with np.load(args.input,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.action=='prepare':
        frame=dict(ids=np.asarray(a['val_ids']).astype(str),xy=a['val_xy'],targets=np.asarray(TARGETS))
        r=build_injections(f'cv_b{args.buffer_km}_fold{args.fold}',frame=frame)
        a.update({k:r[k] for k in ('deltas','positive_mask','negative_mask','case_ids')})
        np.savez_compressed(args.output,**a)
        args.output.with_suffix('.json').write_text(json.dumps(r['metadata'],indent=2)+'\n')
    elif args.action=='core-predict':
        from geopr_vae.pipeline import load,predict
        from geopr_vae.preprocessing import transform
        with np.load(args.query,allow_pickle=False) as z:q={k:z[k] for k in z.files}
        np.testing.assert_array_equal(np.asarray(q['ids']).astype(str),np.asarray(a['val_ids']).astype(str))
        bundle=load(args.model);pp=transform(q,bundle['preprocessing']);scores=[]
        for delta in np.concatenate((np.zeros_like(a['deltas'][:1]),a['deltas'])):
            changed=dict(q,log10=np.asarray(q['log10'])+delta*pp['sigma'])
            prediction,_=predict(bundle,changed)
            scores.append(prediction['raw_scores'][:,0])
        np.savez_compressed(args.output,native_scores=np.stack(scores),query_ids=np.asarray(q['ids']).astype(str))
    else:
        with np.load(args.predictions,allow_pickle=False) as z:
            scores=z['native_scores'];ids=z['query_ids']
        np.testing.assert_array_equal(ids,a['val_ids'])
        if scores.shape!=(145,len(ids)):raise ValueError('Expected 145 complete native score fields')
        rows=[]
        for i,case in enumerate(a['case_ids']):
            p=a['positive_mask'][i];n=a['negative_mask'][i];row={'case_id':str(case)}
            for prefix,s in [('after',scores[i+1]),('delta',scores[i+1]-scores[0])]:
                row.update({prefix+'_'+k:v for k,v in binary_metrics(s,p,n).items()})
                row[prefix+'_core_recall_top10']=recall_budget(s,p,ids)
            rows.append(row)
        args.output.write_text(json.dumps({'cases':rows,'fold_mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='case_id'}},indent=2)+'\n')

if __name__=='__main__':main()
