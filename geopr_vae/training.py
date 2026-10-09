"""Train-only nuisance learning from frozen M40 pseudo-null/intervention math."""
import time
import numpy as np
import torch
from .recovery import ResidualRecoveryHead, normalize_config
from .pseudo_training import _bank, _rng

def nuisance_loss(n0,n1,base_null,delta,config):
    """Signed nuisance targets only in pseudo-null; preservation penalizes its change."""
    residual0=base_null-n0;dn=n1-n0
    energy=delta.square().sum().clamp_min(1e-8);mask=delta>0
    on=(dn.square()*mask).sum()/energy;off=(dn.square()*(~mask)).sum()/energy
    equiv=on+config['spill_weight']*off
    null=residual0.square().mean();sparse=residual0.abs().mean()
    null_term=0. if config['no_null'] else config['null_weight']*null+config['sparse_weight']*sparse
    equiv_term=0. if config['no_equiv'] else config['eq_weight']*equiv
    return dict(total_loss=null_term+equiv_term,null_loss=null,sparse_loss=sparse,
        equiv_on=on,equiv_off=off,equiv_loss=equiv,
        nuisance_null_rms=n0.square().mean().sqrt(),nuisance_change_rms=dn.square().mean().sqrt(),
        null_correction_rms=null.sqrt(),mean_null_correction=residual0.mean())

def train_residual_head(train_x,train_c,train_xy,train_reference,seed,config=None,epochs=300):
    """Return head,history; never accepts cal/query/B/A/file dictionaries."""
    cfg=normalize_config(config);torch.set_num_threads(1)
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Valid Python integer seed required')
    if type(epochs) is not int or epochs<1:raise ValueError('Positive integer epochs')
    x,c,xy,ref=map(np.asarray,(train_x,train_c,train_xy,train_reference));n=len(x)
    if n<24 or x.shape!=(n,12) or ref.shape!=x.shape or c.shape!=(n,19) or xy.shape!=(n,2):raise ValueError('Explicit train arrays N12/N19/N2/N12 required')
    if not all(np.isfinite(a).all() for a in (x,c,xy,ref)):raise ValueError('Finite training arrays required')
    start=time.perf_counter();scale,f0,f1,delta,metadata=_bank(x,c,xy,ref,seed,cfg)
    prep_seconds=time.perf_counter()-start
    model=ResidualRecoveryHead(scale,cfg,seed);model.train()
    t0=torch.as_tensor(f0,dtype=torch.float32);t1=torch.as_tensor(f1,dtype=torch.float32);td=torch.as_tensor(delta,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=_rng(seed,'M41_nuisance_bank_order_v1');order=[];history=[];fit_start=time.perf_counter()
    for epoch in range(1,epochs+1):
        if not order:order=rng.permutation(cfg['bank_fields']).tolist()
        case=int(order.pop());optimizer.zero_grad(set_to_none=True)
        n0=model.nuisance_from_features(t0[case]);n1=model.nuisance_from_features(t1[case])
        terms=nuisance_loss(n0,n1,t0[case,...,:12],td[case],cfg);loss=terms['total_loss']
        if not isinstance(loss,torch.Tensor) or not torch.isfinite(loss):raise FloatingPointError('Invalid/nonfinite M41 objective')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        history.append(dict(epoch=epoch,bank_index=case,gradient_norm=float(gradient),**{k:float(v.detach()) for k,v in terms.items()}))
    metadata.update(seed=seed,epochs=epochs,preparation_seconds=prep_seconds,fit_seconds=time.perf_counter()-fit_start,
        parameter_count=sum(p.numel() for p in model.parameters()),
        feature_mode=cfg['feature_mode'],initialization='Zero final nuisance weights/bias: exact unit residual skip before learning',
        skip='Unit input-residual skip; signed nuisance has no fixed gain ceiling or positivity clipping',
        objective='Pseudo-null residual is nuisance target; nuisance must change little under independent positive pseudo-intervention',
        training_truth_access=False,evaluation_arrays_accepted=False,
        paired_teacher='Teacher reference value fixed within each pseudo-pair; full teacher is re-inferred after real evaluation intervention',
        invariance='Neighbor contrast features ignore spatially constant element shifts only with unchanged teacher, conditions, coordinates and whole query membership',
        limitation='Training residuals may include enrichment or teacher bias; low-frequency noise can survive the common-mode-preserving nuisance parameterization')
    model.training_metadata=metadata;model.eval();return model,history
