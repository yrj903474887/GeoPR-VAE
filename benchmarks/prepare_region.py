"""Prepare the five spatial folds and separate full-region deployment inputs.

Input NPZ uses log10[N,12], terrain[N,3], xy[N,2] projected metres, groups[N]
and unique string ids[N]. Optional valid[N] flags terrain-quality failures.
Coordinate partitions and buffers use the full spatial frame, then invalid
rows are excluded from all roles. This order preserves the study protocol.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from geopr_vae.preprocessing import validate_data,regional_split,fit_preprocessor,transform
from .region_partition import coordinate_partition

def prepare(raw,mode='cv',fold=1,buffer_m=4000):
    xy=np.asarray(raw['xy'],float);ids=np.asarray(raw['ids']).astype(str)
    if xy.shape!=(len(ids),2) or not np.isfinite(xy).all() or len(np.unique(ids))!=len(ids):
        raise ValueError('A complete finite coordinate frame with unique IDs is required')
    good=np.asarray(raw.get('valid',np.ones(len(ids),bool)),bool)
    good=good & np.isfinite(raw['log10']).all(1) & np.isfinite(raw['terrain']).all(1)
    keep=np.flatnonzero(good);full_to_local=np.full(len(ids),-1,int);full_to_local[keep]=np.arange(len(keep))
    d=validate_data({k:np.asarray(raw[k])[keep] for k in ('log10','terrain','xy','groups','ids')})
    if mode=='global':
        tr,cal=regional_split(d['ids'],d['xy']);query=np.arange(len(keep))
        seed=dict(reference_seed=20260929,head_seed=20260929,patch_seed=2027101004)
        partition=None
    else:
        if fold not in range(1,6) or buffer_m not in (4000,8000):raise ValueError('Expected fold 1..5 and 4/8 km buffer')
        labels,partition=coordinate_partition(xy,5,ids)
        centroids={f:xy[labels==f].mean(0) for f in range(1,6)}
        devfold=sorted([f for f in range(1,6) if f!=fold],key=lambda f:(-np.linalg.norm(centroids[f]-centroids[fold]),f))[0]
        test=np.flatnonzero(labels==fold);dev_all=np.flatnonzero(labels==devfold)
        candidates=np.flatnonzero((labels!=fold)&(labels!=devfold))
        dev=dev_all[cKDTree(xy[test]).query(xy[dev_all])[0]>buffer_m]
        train=candidates[cKDTree(xy[np.r_[test,dev_all]]).query(xy[candidates])[0]>buffer_m]
        tr,cal,query=[full_to_local[i[good[i]]] for i in (train,dev,test)]
        seed=dict(reference_seed=2026091202+104729*(fold-1),head_seed=2026410101+104729*fold,
                  patch_seed=2026101001+104729*(fold-1))
    if min(len(tr),len(cal),len(query))<2:raise ValueError('Too few observations in a role')
    state=fit_preprocessor(d,tr);pp=transform(d,state);a={}
    for role,ix in [('train',tr),('cal',cal),('val',query)]:
        for target,source in [('x','r'),('c','F'),('xy','xy'),('ids','ids')]:a[role+'_'+target]=pp[source][ix]
    core=dict(d,train=tr,calibration=cal)
    q={k:v[query] for k,v in d.items()}
    info=dict(mode=mode,fold=fold if mode=='cv' else None,buffer_m=buffer_m if mode=='cv' else None,
              train_n=len(tr),cal_n=len(cal),query_n=len(query),input_n=len(ids),valid_n=len(keep),
              seeds=seed,partition=partition)
    return a,core,q,info

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=['cv','global'],default='cv');ap.add_argument('--fold',type=int,default=1)
    ap.add_argument('--buffer-km',type=int,choices=[4,8],default=4)
    args=ap.parse_args()
    with np.load(args.input,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    a,core,q,info=prepare(d,args.mode,args.fold,args.buffer_km*1000)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,obj in [('patch_input',a),('core_train',core),('core_query',q)]:np.savez_compressed(args.output/(name+'.npz'),**obj)
    (args.output/'settings.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':main()
