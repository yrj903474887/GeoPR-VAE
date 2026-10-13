"""Unit residual skip with learned signed nuisance subtraction.

The default nuisance branch uses differences between the point residual and
two neighborhood means. With teacher values fixed, an elementwise constant
shift of the complete query field leaves those differences invariant. The
full teacher-plus-head system is target-visible and need not share that
invariance when the frozen teacher is re-inferred after an intervention.
"""
import numpy as np
import torch
from torch import nn
from .spatial import neighborhood_matrices, features_for_fields

DEFAULT_CONFIG=dict(width=32,eq_weight=32.,null_weight=1.,sparse_weight=.001,
    spill_weight=.1,learning_rate=.001,weight_decay=.0001,gradient_clip=5.,
    bank_fields=24,max_training_nodes=512,neighbor_lengths_m=(3000.,9000.),
    spectral_length_m=6000.,no_spatial=False,no_equiv=False,no_null=False,
    feature_mode='neighbor_contrast')

def normalize_config(config=None):
    d=dict(DEFAULT_CONFIG)
    if config:
        unknown=set(config)-set(d)
        if unknown:raise ValueError('Unknown recovery configuration '+str(unknown))
        d.update(config)
    for k in ('width','bank_fields','max_training_nodes'):
        if int(d[k])!=d[k]:raise ValueError('Integer '+k)
        d[k]=int(d[k])
    if d['width']<4 or d['bank_fields']<4 or d['bank_fields']%2 or not 24<=d['max_training_nodes']<=512:raise ValueError('Invalid width/bank/subgraph count')
    for k in ('eq_weight','null_weight','sparse_weight','spill_weight','learning_rate','weight_decay','gradient_clip','spectral_length_m'):
        if not np.isfinite(d[k]) or d[k]<0:raise ValueError(k)
    if d['spectral_length_m']<=0 or len(d['neighbor_lengths_m'])!=2 or not np.isfinite(d['neighbor_lengths_m']).all() or min(d['neighbor_lengths_m'])<=0:raise ValueError('Positive graph scales')
    if d['feature_mode'] not in ('neighbor_contrast','raw_residual'):raise ValueError('feature_mode')
    if d['no_null'] and d['no_equiv']:raise ValueError('No active training objective')
    return d

def nuisance_features(features,feature_mode='neighbor_contrast'):
    """Retain57 dimensions, suppress chemical common-mode information by design."""
    if features.shape[-1]!=57:raise ValueError('Expected57 original features')
    if feature_mode=='raw_residual':return features
    if feature_mode!='neighbor_contrast':raise ValueError('Unknown feature mode')
    e,m3,m9=features[...,:12],features[...,12:24],features[...,24:36]
    has3=(features[...,55:56]>0).to(features.dtype);has9=(features[...,56:57]>0).to(features.dtype)
    # A missing neighborhood contributes no difference, rather than comparing
    # against an artificial zero that would violate common-mode invariance.
    return torch.cat(((e-m3)*has3,(e-m9)*has9,(m3-m9)*has3*has9,features[...,36:]),dim=-1)

class ResidualRecoveryHead(nn.Module):
    def __init__(self,noise_scale,config=None,seed=0):
        super().__init__();self.config=normalize_config(config);self.seed=int(seed)
        s=np.asarray(noise_scale,dtype=np.float32)
        if s.shape!=(12,) or not np.isfinite(s).all() or np.any(s<=0):raise ValueError('Positive training scale[12] required')
        self.register_buffer('noise_scale',torch.as_tensor(s));w=self.config['width']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network=nn.Sequential(nn.Linear(57,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,12))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.training_metadata={}
    def nuisance_from_features(self,features):
        return self.network(nuisance_features(features,self.config['feature_mode']))
    def forward_features(self,features):
        return features[...,:12]-self.nuisance_from_features(features)
    def predict(self,residual,c,xy,batch_size=16384):
        original=np.asarray(residual);single=original.ndim==2
        if type(batch_size) is not int or batch_size<1:raise ValueError('Positive integer batch_size')
        graphs,effective=neighborhood_matrices(xy,self.config['neighbor_lengths_m'])
        features=features_for_fields(original,c,self.noise_scale.cpu().numpy(),graphs,effective,self.config['no_spatial'])
        shape=features.shape[:-1]+(12,);flat=features.reshape(-1,57);pieces=[]
        mode=self.training;self.eval()
        with torch.no_grad():
            for start in range(0,len(flat),batch_size):
                f=torch.as_tensor(flat[start:start+batch_size],dtype=torch.float32)
                pieces.append((self.nuisance_from_features(f)*self.noise_scale).cpu().numpy())
        self.train(mode);nuisance=np.concatenate(pieces).reshape(shape)
        if single:nuisance=nuisance[0]
        # The unit skip itself uses the caller's observed residual exactly;
        # network nuisance is predicted in float32 and converted as necessary.
        correction=original-nuisance
        if not np.isfinite(correction).all():raise FloatingPointError('Nonfinite nuisance subtraction')
        return dict(correction=correction,score_channels=correction.copy(),nuisance_component=nuisance,
            noise_correction=-nuisance,signed_skip_component=original.copy(),gate=np.ones_like(correction),
            positive_component=np.zeros_like(correction),
            diagnostics=dict(parameter_count=sum(p.numel() for p in self.parameters()),
                feature_mode=self.config['feature_mode'],no_spatial=self.config['no_spatial'],
                neighbor_lengths_m=list(self.config['neighbor_lengths_m']),
                effective_neighbors_min=effective.min(0).tolist(),effective_neighbors_max=effective.max(0).tolist(),
                algebra='correction = residual - nuisance_component; D = teacher + correction',
                compatibility_fields='gate=1 denotes fixed skip, positive_component=0 is a storage placeholder; signed_skip_component=residual',
                invariance='Conditional on unchanged teacher, conditions and query geometry: neighbor_contrast nuisance ignores per-element constant residual shifts; full teacher re-inference may change this response'))
    def state(self):
        return dict(format='M41-unit-skip-nuisance-head-v1',config=self.config,seed=self.seed,
            noise_scale=self.noise_scale.cpu().numpy().tolist(),state_dict=self.state_dict(),training_metadata=self.training_metadata)
def restore(state):
    if state.get('format')!='M41-unit-skip-nuisance-head-v1':raise ValueError('Invalid recovery checkpoint')
    head=ResidualRecoveryHead(state['noise_scale'],state['config'],state['seed'])
    head.load_state_dict(state['state_dict'],strict=True);head.training_metadata=state.get('training_metadata',{})
    return head
