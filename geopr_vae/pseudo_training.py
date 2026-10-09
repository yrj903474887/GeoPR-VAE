"""Training-only pseudo-null and intervention bank."""
import hashlib
import numpy as np
from .spatial import neighborhood_matrices, features_for_fields

def _rng(seed,namespace):
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Integer seed required')
    actual=int.from_bytes(hashlib.sha256((namespace+'|'+str(seed)).encode()).digest()[:8],'big')
    return np.random.default_rng(actual)


def _robust_scale(x):
    med=np.median(x,axis=0);mad=1.4826*np.median(np.abs(x-med),axis=0)
    sd=np.std(x,axis=0,ddof=1)
    return np.maximum(np.where(mad>1e-8,mad,sd),.1)


def _spectral_basis(xy,length):
    if len(xy)>512:raise ValueError('Dense eigensystem is limited to512 training nodes')
    delta=xy[:,None]-xy[None,:];d2=np.square(delta).sum(-1)
    w=np.exp(-.5*d2/(length*length));w[d2>(3*length)**2]=0.;np.fill_diagonal(w,0)
    lap=np.diag(w.sum(1))-w
    eigenvalues,u=np.linalg.eigh(lap)
    return lap,eigenvalues,u


def _intervention(xy,observed_scale,rng):
    """Continuous Gaussian mixtures, fresh centers, Bernoulli support, new doses.

    Distinct from evaluation's six maximin centers, three fixed polynomial
    kernels, support2/5/8/12 and amplitude.25/.5 templates. Training amplitudes
    are relative to robust observed train_x scale, not evaluation truth.
    """
    field=np.zeros((len(xy),12));bumps=int(rng.integers(1,4));details=[]
    for _ in range(bumps):
        radii=np.exp(rng.uniform(np.log(2000.),np.log(18000.),2));angle=rng.uniform(0,np.pi)
        anchor=int(rng.integers(len(xy)));center=xy[anchor]+rng.uniform(-.2,.2,2)*radii
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local=(xy-center)@rot;q=np.square(local/radii).sum(1)
        weights=np.where(q<9.,np.exp(-.5*q),0.)
        probability=rng.uniform(.1,.8);support=rng.random(12)<probability
        if not support.any():support[rng.integers(12)]=True
        direction=np.where(support,np.exp(rng.normal(0,.5,12)),0.);direction/=np.linalg.norm(direction)
        field+=weights[:,None]*direction[None]
        details.append(dict(anchor=anchor,center_xy_m=center.tolist(),radii_m=radii.tolist(),
            angle_radians=float(angle),support_size=int(support.sum())))
    peak=np.linalg.norm(field,axis=1).max()
    if peak<=0:raise RuntimeError('Empty independently generated training intervention')
    amplitude=float(np.exp(rng.uniform(np.log(.04),np.log(.9))))
    field*=amplitude/peak
    return field*observed_scale,dict(amplitude=amplitude,bumps=details,
        peak_observed_scale_L2=amplitude,template='continuous anisotropic Gaussian mixture / Bernoulli element support')


def _bank(train_x,train_c,train_xy,train_reference,seed,config):
    n=len(train_x)
    rng_subset=_rng(seed,'M40_training_subset_v1')
    indices=np.arange(n) if n<=config['max_training_nodes'] else np.sort(rng_subset.choice(n,config['max_training_nodes'],replace=False))
    residual=np.asarray(train_x,float)-np.asarray(train_reference,float)
    noise_scale=_robust_scale(residual);observed_scale=_robust_scale(np.asarray(train_x,float))
    xy=np.asarray(train_xy,float)[indices];c=np.asarray(train_c,np.float32)[indices]
    e=residual[indices]/noise_scale
    lap,eigenvalues,u=_spectral_basis(xy,config['spectral_length_m'])
    coeff=u.T@e;gram=e.T@e;energy=e.T@lap@e
    rn=_rng(seed,'M40_pseudo_null_v1');rd=_rng(seed,'M40_positive_interventions_v1')
    nulls=[];deltas=[];meta=[];gram_error=0.;energy_error=0.
    for pair in range(config['bank_fields']//2):
        # Alternating laws preserve spectral second-order structure, with half
        # the pairs retaining the exact observed tail values under global sign.
        if pair%2==0:
            signs=rn.choice([-1.,1.],size=len(xy));null=u@(signs[:,None]*coeff);kind='joint_graph_spectral_sign'
        else:
            null=e*float(rn.choice([-1.,1.]));kind='whole_field_sign'
        gram_error=max(gram_error,float(np.max(np.abs(null.T@null-gram))))
        energy_error=max(energy_error,float(np.max(np.abs(null.T@lap@null-energy))))
        for polarity in (1.,-1.):
            nulls.append(polarity*null)
            delta,detail=_intervention(xy,observed_scale,rd)
            deltas.append(delta/noise_scale)
            meta.append(dict(null_kind=kind,antithetic_polarity=polarity,**detail))
    nulls=np.stack(nulls);deltas=np.stack(deltas)
    graphs,effective=neighborhood_matrices(xy,config['neighbor_lengths_m'])
    f0=features_for_fields(nulls*noise_scale,c,noise_scale,graphs,effective,config['no_spatial'])
    f1=features_for_fields((nulls+deltas)*noise_scale,c,noise_scale,graphs,effective,config['no_spatial'])
    info=dict(original_training_n=n,head_training_n=len(indices),training_subset_indices=indices.tolist(),
        subset_rule='uniform without replacement using dedicated training seed; no chemistry/score-based selection',
        bank_fields=config['bank_fields'],null_source='training teacher residuals only',
        teacher_residuals_in_sample=True,noise_scale=noise_scale.tolist(),observed_scale=observed_scale.tolist(),
        antithetic_null_mean_max_abs=float(np.max(np.abs(nulls.mean(0)))),
        null_gram_max_abs_error=gram_error,null_graph_energy_max_abs_error=energy_error,
        spectral_max_nodes=512,spectral_actual_nodes=len(xy),
        preserved='uncentered across-element Gram and combinatorial-Laplacian quadratic form on the selected standardized-residual subgraph',
        limitations='spectral law need not preserve tail shape or exact distance-binned variogram; whole-field sign law retains empirical values; nuisance bank may include real enrichment/background bias',
        intervention_metadata=meta,training_truth_access=False,evaluation_arrays_accepted=False)
    return noise_scale,f0,f1,deltas,info

