"""Role-isolated 10x10 IDW fields, bilinearly resized to 32x32 patches."""
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
PATCH_CONFIG = dict(native_grid=10, spacing_m=2000., resize=32, idw_k=12,
                    idw_power=2., scaling='train_minmax_01')
class PatchGeometry:
    def __init__(self,xy,centers=None):
        xy=np.asarray(xy,float)
        centers=xy if centers is None else np.asarray(centers,float)
        v=(np.arange(10)-4.5)*2000.
        xx,yy=np.meshgrid(v,v)
        self.offset=np.column_stack((xx.ravel(),yy.ravel()))
        queries=(centers[:,None,:]+self.offset[None]).reshape(-1,2)
        dist,ind=cKDTree(xy).query(queries,k=min(12,len(xy)))
        if ind.ndim==1:ind=ind[:,None];dist=dist[:,None]
        weight=1/np.maximum(dist,1.e-8)**2
        exact=dist[:,0]<1.e-7
        weight[exact]=0;weight[exact,0]=1
        weight/=weight.sum(1,keepdims=True)
        self.indices=ind.reshape(len(centers),100,-1)
        self.weights=weight.reshape(len(centers),100,-1).astype(np.float32)
        self.n=len(centers)
        self.max_nearest_m=float(dist[:,0].max())
        self.p95_nearest_m=float(np.quantile(dist[:,0],.95))
    def patches(self,values,low,span,select=None):
        """Training-fitted affine scaling; query extrema never affect scaling."""
        values=np.asarray(values,np.float32)
        ind=self.indices if select is None else self.indices[select]
        w=self.weights if select is None else self.weights[select]
        z=(values-low)/span
        # Preserve interpolation of concentrations, then apply fixed clipping.
        native=np.einsum('nki,nkic->nkc',w,z[ind],optimize=True)
        native=np.clip(native,0.,1.).reshape(-1,10,10,values.shape[1]).transpose(0,3,1,2).copy()
        with torch.inference_mode():
            result=F.interpolate(torch.from_numpy(native),size=(32,32),mode='bilinear',align_corners=False).numpy()
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite patch')
        return result
