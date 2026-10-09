"""Leave-self-out recovery graphs and field features."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

def neighborhood_matrices(xy,lengths=(3000.,9000.)):
    """Leave-self-out Gaussian radius graphs; sparse and safe for regional N."""
    xy=np.asarray(xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not len(xy) or not np.isfinite(xy).all():raise ValueError('xy must be nonempty finite[N,2]')
    n=len(xy);tree=cKDTree(xy);out=[];effective=[]
    for h in lengths:
        ii=[];jj=[];vv=[]
        for i,neighbors in enumerate(tree.query_ball_point(xy,3.*h)):
            neighbors=np.asarray([j for j in neighbors if j!=i],dtype=int)
            if not len(neighbors):continue
            w=np.exp(-.5*np.square((xy[neighbors]-xy[i])/h).sum(1));w/=w.sum()
            ii.extend([i]*len(neighbors));jj.extend(neighbors.tolist());vv.extend(w.tolist())
        w=csr_matrix((vv,(ii,jj)),shape=(n,n),dtype=np.float32)
        squared=np.asarray(w.multiply(w).sum(1)).ravel()
        eff=np.divide(1.,squared,out=np.zeros(n),where=squared>0)
        out.append(w);effective.append(eff)
    return out,np.column_stack(effective).astype(np.float32)


def features_for_fields(residual,c,noise_scale,graphs,effective,no_spatial=False):
    r=np.asarray(residual,dtype=np.float32);c=np.asarray(c,dtype=np.float32)
    if r.ndim==2:r=r[None]
    if r.ndim!=3 or r.shape[-1]!=12 or c.shape!=(r.shape[1],19):raise ValueError('Expected residual[N,12] or[K,N,12], c[N,19]')
    if not np.isfinite(r).all() or not np.isfinite(c).all():raise ValueError('Finite features required')
    k,n,_=r.shape;e=r/np.asarray(noise_scale,dtype=np.float32)[None,None,:]
    neigh=[]
    for w in graphs:
        mean=(w@e.transpose(1,0,2).reshape(n,k*12)).reshape(n,k,12).transpose(1,0,2)
        neigh.append(np.zeros_like(mean) if no_spatial else mean)
    count=np.zeros_like(effective) if no_spatial else np.log1p(effective)/5.
    return np.concatenate([e,*neigh,np.broadcast_to(c,(k,n,19)),np.broadcast_to(count,(k,n,2))],axis=-1)

