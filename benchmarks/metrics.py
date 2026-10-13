"""Recovery, detection and paired base-level uncertainty metrics."""
import numpy as np
import math
BUDGETS=(.01,.02,.05,.10,.20)
def ranked_recall(scores,ids,positive,budget):
    y=np.asarray(positive,dtype=bool);s=np.asarray(scores,float)
    if s.shape!=y.shape or np.asarray(ids).shape!=s.shape or not y.any():raise ValueError('Invalid recall inputs')
    k=int(np.ceil(len(s)*budget))
    order=np.lexsort((np.asarray(ids).astype(str),-s))
    return float(y[order[:k]].sum()/y.sum()),k

def roc_diagnostics(scores,positive):
    """Complete tied-score blocks; AP is non-interpolated stepwise recall integral."""
    s=np.asarray(scores,float);y=np.asarray(positive,bool)
    if s.shape!=y.shape or not y.any() or y.all():raise ValueError('Both classes required')
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(yy)[ends];fp=np.cumsum(~yy)[ends]
    tpr=np.r_[0.,tp/y.sum()];fpr=np.r_[0.,fp/(~y).sum()]
    auroc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])*.5))
    ap=float(np.sum(np.diff(np.r_[0.,tp/y.sum()])*(tp/(tp+fp))))
    best=float(tpr[fpr<=.05+1e-12].max())
    return dict(auroc=auroc,average_precision=ap,tpr_fpr05=best,
                fpr=fpr,tpr=tpr,thresholds=np.r_[np.inf,ss[ends]])

def paired_base_bootstrap(a,b,direction,seed=2026399001,repetitions=10000):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2:raise ValueError('Aligned independent base means required')
    delta=a-b;benefit=direction*delta
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(a),(repetitions,len(a)))
    raw_ci=np.quantile(delta[indices].mean(1),[.025,.975])
    benefit_ci=np.quantile(benefit[indices].mean(1),[.025,.975])
    mean=float(benefit.mean())
    point='numerically_tied' if abs(mean)<=1e-12 else ('numerically_ahead' if mean>0 else 'numerically_behind')
    evidence='ahead_interval' if benefit_ci[0]>0 else ('behind_interval' if benefit_ci[1]<0 else 'difference_unresolved')
    return dict(n_base=len(a),mean_difference=float(delta.mean()),difference_CI95=raw_ci,
        benefit_CI95=benefit_ci,point_status=point,interval_status=evidence,
        bootstrap_repetitions=repetitions,bootstrap_seed=seed,
        interpretation='paired base-level percentile bootstrap; descriptive, no equivalence test or familywise claim')
def evaluate_arrays(a,reference,denoised,scored):
    delta=np.asarray(a['deltas'],float);receiver=np.asarray(a['receivers'],bool)
    x0=np.asarray(a['val_x'],float);bg=np.asarray(a['bg'],float)
    if x0.ndim!=2 or x0.shape[1]!=12 or bg.shape!=x0.shape:raise ValueError('B2 field shapes')
    if delta.ndim!=3 or delta.shape[1:]!=x0.shape or not len(delta):raise ValueError('Injection shape')
    if not all(np.isfinite(v).all() for v in [x0,bg,delta]):raise ValueError('Finite evaluation truth/input required')
    ids=np.asarray(a['val_ids']).astype(str)
    if ids.shape!=(len(x0),) or len(np.unique(ids))!=len(ids):raise ValueError('Unique query identities required')
    if receiver.shape!=delta.shape[:2] or not np.array_equal(receiver,np.any(delta!=0,axis=-1)):raise ValueError('Receiver mask differs from injection support')
    shape=(len(delta)+1,len(x0),12)
    ref=np.asarray(reference,float).reshape(shape);den=np.asarray(denoised,float).reshape(shape)
    scores=np.asarray(scored['score'],float);threshold=float(scored['threshold'])
    cal_scores=np.asarray(scored['cal_score'],float)
    if not np.isfinite(threshold) or cal_scores.ndim!=1 or not len(cal_scores) or not np.isfinite(cal_scores).all():raise ValueError('Finite threshold and calibration scores required')
    assert scores.shape==shape[:2] and receiver.shape==delta.shape[:2]
    assert receiver.any(1).all() and (~receiver).any(1).all()
    assert np.isfinite(ref).all() and np.isfinite(den).all() and np.isfinite(scores).all()
    target=bg[None]+delta;err=den[1:]-target;energy=np.square(delta).sum((1,2))
    assert np.all(energy>0)
    gain=np.sum(delta*(den[1:]-den[0]),axis=(1,2))/energy
    k=len(delta);n=len(x0)
    case=dict(denoising_mae=np.abs(err).mean((1,2)),denoising_rmse=np.sqrt(np.square(err).mean((1,2))),
        denoising_mse=np.square(err).mean((1,2)),
        receiver_mae=np.array([np.abs(err[i,y]).mean() for i,y in enumerate(receiver)]),
        receiver_rmse=np.array([np.sqrt(np.square(err[i,y]).mean()) for i,y in enumerate(receiver)]),
        native_reference_mae=np.abs(ref[1:]-target).mean((1,2)),native_background_mae=np.abs(ref[1:]-bg).mean((1,2)),
        amplitude_gain=gain,amplitude_gain_abs_error=np.abs(gain-1),
        fixed_threshold_receiver_detection=np.array([(scores[i+1,y]>threshold).mean() for i,y in enumerate(receiver)]),
        injected_nonreceiver_fpr=np.array([(scores[i+1,~y]>threshold).mean() for i,y in enumerate(receiver)]),
        receiver_fraction=receiver.mean(1))
    for budget in BUDGETS:
        label=f'{int(round(budget*100)):02d}'
        before=np.array([ranked_recall(scores[0],a['val_ids'],y,budget)[0] for y in receiver])
        after=np.array([ranked_recall(scores[i+1],a['val_ids'],y,budget)[0] for i,y in enumerate(receiver)])
        for tag,v in [('before',before),('after',after),('net',after-before)]:case[f'recall_p{label}_{tag}']=v
    for tag in ['before','after','net']:case['top5_'+tag]=case['recall_p05_'+tag]
    for key in ['tpr_fpr05','auroc','average_precision']:case[key]=np.empty(k);case[key+'_before']=np.empty(k)
    rocs=[]
    for i,y in enumerate(receiver):
        before=roc_diagnostics(scores[0],y);after=roc_diagnostics(scores[i+1],y)
        for key in ['tpr_fpr05','auroc','average_precision']:case[key][i]=after[key];case[key+'_before'][i]=before[key]
        rocs.append((after['fpr'],after['tpr'],after['thresholds']))
    case['tpr_before']=case['tpr_fpr05_before'];case['tpr_net']=case['tpr_fpr05']-case['tpr_before']
    summary={key:float(values.mean()) for key,values in case.items()}
    summary.update(unmodified_E_FPR=float((scores[0]>threshold).mean()),
        no_injection_denoising_mae=float(np.abs(den[0]-bg).mean()),no_injection_denoising_rmse=float(np.sqrt(np.square(den[0]-bg).mean())),
        n_cases=k,n_query=n,n_threshold=len(scored.get('threshold_indices',cal_scores)),calibration_threshold=threshold)
    names=tuple(case);offsets=np.r_[0,np.cumsum([len(v[0]) for v in rocs])]
    arrays=dict(case_metrics=np.column_stack([case[key] for key in names]),case_metric_names=np.asarray(names),
        scores=scores,cal_scores=np.asarray(scored['cal_score']),threshold=np.asarray(threshold),
        roc_offsets=offsets,roc_fpr=np.concatenate([v[0] for v in rocs]),roc_tpr=np.concatenate([v[1] for v in rocs]),
        roc_thresholds=np.concatenate([v[2] for v in rocs]))
    return summary,arrays

def readout(x,ref,config):
    kind=config['kind'];r=np.asarray(x)-ref
    if kind=='native':return np.asarray(ref)
    if kind=='positive':return ref+np.maximum(r-float(config['tau']),0)
    if kind=='linear':return ref+float(config['alpha'])*r
    if kind=='leaky':
        alpha=float(config['alpha']);return ref+alpha*r+(1-alpha)*np.maximum(r-float(config['tau']),0)
    raise ValueError('Unknown common readout')
def binary_metrics(score,positive,negative):
    score=np.asarray(score,float);p=np.asarray(positive,bool);n=np.asarray(negative,bool)
    if (p&n).any() or not p.any() or not n.any() or not np.isfinite(score).all():raise ValueError('Invalid ROC population')
    keep=p|n;s=score[keep];y=p[keep];order=np.argsort(-s,kind='stable');s=s[order];y=y[order]
    last=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[last].astype(float);fp=(last+1)-tp
    tpr=np.r_[0,tp/p.sum()];fpr=np.r_[0,fp/n.sum()]
    integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    auc=float(integrate(tpr,fpr))
    ap=float(np.sum(np.diff(tpr)*(tp/(tp+fp))))
    prevalence=float(p.sum()/keep.sum())
    return dict(auc=auc,ap=ap,ap_minus_prevalence=ap-prevalence,prevalence=prevalence)

def recall_budget(score,positive,ids):
    score=np.asarray(score,float);ids=np.asarray(ids).astype(str);positive=np.asarray(positive,bool)
    if len(np.unique(ids))!=len(ids):raise ValueError('Sample IDs not unique')
    k=math.ceil(.10*len(score));order=np.lexsort((ids,-score));selected=order[:k]
    return float(positive[selected].sum()/positive.sum())
