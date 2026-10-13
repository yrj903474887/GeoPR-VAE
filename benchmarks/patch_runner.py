"""Train and apply the spatial convolutional comparisons to user-supplied arrays.

Run ``python -m benchmarks.patch_runner --help`` from the repository root.
The input chemistry must already use training-fitted standardization. Each
role builds its own interpolation graph; calibration halves are isolated.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from geopr_vae.scoring import split_calibration
from . import models
from .patches import PatchGeometry, PATCH_CONFIG

METHODS=('VAE_patch','VAE_GAN','VAE_WGAN','VAE_WGANGP','GANomaly','Skip_GANomaly','f_AnoGAN','Geo_Hgan')

def validate(a):
    for role in ('train','cal','val'):
        x=np.asarray(a[role+'_x']); xy=np.asarray(a[role+'_xy']); ids=np.asarray(a[role+'_ids']).astype(str)
        if x.ndim!=2 or x.shape[1]!=12 or xy.shape!=(len(x),2) or ids.shape!=(len(x),):
            raise ValueError('Expected x[N,12], xy[N,2], ids[N] for '+role)
        if not len(x) or not np.isfinite(x).all() or not np.isfinite(xy).all() or len(np.unique(ids))!=len(ids):
            raise ValueError('Invalid '+role+' arrays')
    if set(a['train_ids'].astype(str)) & set(a['cal_ids'].astype(str)):
        raise ValueError('Training and calibration IDs must be disjoint')

def fit_model(a,method,seed,epochs=100,upstream_dir=None):
    validate(a)
    if method not in METHODS: raise ValueError(method)
    low=np.asarray(a['train_x'].min(0),np.float32)
    span=np.asarray(np.maximum(a['train_x'].max(0)-low,1.e-6),np.float32)
    train=PatchGeometry(a['train_xy']).patches(a['train_x'],low,span)
    if method=='Geo_Hgan':
        from .geohgan import GeoHGAN,GeoHGANConfig
        m=GeoHGAN(GeoHGANConfig(channels=12,seed=int(seed),threads=1,
            epochs_stage1=epochs,epochs_stage2=epochs,upstream_dir=upstream_dir))
        m.fit(train)
    else:
        m=models.fit(train,'VAE' if method=='VAE_patch' else method,
                     seed=int(seed),epochs=int(epochs),threads=1)
    return m,low,span

def center_prediction(model,geometry,values,low,span):
    patches=geometry.patches(values,low,span)
    if isinstance(model,models.ComparisonModel):pred=model.predict(patches,batch_size=64)
    else:pred=model.predict(patches,batch_size=64,return_reconstruction=True)
    midpoint=pred['reconstruction'][:,:,15:17,15:17].mean(axis=(2,3),dtype=np.float64)
    center16=pred['reconstruction'][:,:,16,16]
    residual16=(patches[:,:,16,16].astype(np.float64)-center16.astype(np.float64))*np.asarray(span,np.float64)
    return midpoint*span+low,np.asarray(pred['score'],float),center16*span+low,residual16

def predict_fields(model,a,low,span):
    fi,ti=split_calibration(a['cal_ids'])
    calref=np.empty_like(a['cal_x'],dtype=float); calnative=np.empty(len(calref))
    cal16=np.empty_like(calref); calres16=np.empty_like(calref)
    for ix in (fi,ti):
        calref[ix],calnative[ix],cal16[ix],calres16[ix]=center_prediction(model,PatchGeometry(a['cal_xy'][ix]),a['cal_x'][ix],low,span)
    fields=np.concatenate((a['val_x'][None],a['val_x'][None]+a['deltas'])) if 'deltas' in a else a['val_x'][None]
    geometry=PatchGeometry(a['val_xy']); reference=np.empty_like(fields,dtype=float); native=np.empty(fields.shape[:2])
    for i,field in enumerate(fields):
        reference[i],native[i],center16,residual16=center_prediction(model,geometry,field,low,span)
        if i==0:q16=center16;qr16=residual16
    return dict(reference=reference,cal_reference=calref,native_scores=native,
                query_center_reconstruction=reference[0],cal_center_reconstruction=calref,
                query_signed_residual=a['val_x']-reference[0],cal_signed_residual=a['cal_x']-calref,
                query_center16_reconstruction=q16,cal_center16_reconstruction=cal16,
                query_patch16_signed_residual=qr16,cal_patch16_signed_residual=calres16,
                native_score=native[0],native_cal_score=calnative,low=low,span=span,
                calfit_indices=fi,threshold_indices=ti,
                query_xy=a['val_xy'],cal_xy=a['cal_xy'],query_ids=a['val_ids'],cal_ids=a['cal_ids'])

def checkpoint(model,low,span):
    return dict(method=model.method,config=asdict(model.config),state_dict=model.state_dict(),
                metadata=model.metadata,history=model.history,qa=model.qa,low=low,span=span)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--method',choices=METHODS,required=True); ap.add_argument('--seed',type=int,default=2026101001)
    ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--save-model',action='store_true')
    ap.add_argument('--upstream-dir',help='Geo-Hgan author checkout; required only for Geo_Hgan')
    args=ap.parse_args()
    with np.load(args.input,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    model,low,span=fit_model(a,args.method,args.seed,args.epochs,args.upstream_dir); p=predict_fields(model,a,low,span)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'predictions.npz',**p)
    model_cfg=model.config
    (args.output/'settings.json').write_text(json.dumps(dict(method=args.method,seed=args.seed,epochs=args.epochs,
        model=asdict(model_cfg),patch=PATCH_CONFIG,controlled_readout='mean of four center pixels',
        regional_readout='matched patch residual at pixel 16,16'),indent=2)+'\n')
    if args.save_model:
        if args.method=='Geo_Hgan':model.save(args.output/'model.pt')
        else:torch.save(checkpoint(model,low,span),args.output/'model.pt')

if __name__=='__main__':main()
