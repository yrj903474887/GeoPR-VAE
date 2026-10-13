"""Train-only standardization, spatial folds and controlled weak probes."""
from __future__ import annotations
import hashlib
import numpy as np
from . import analytic_fields as old
from . import noise as helper
FAMILIES=old.FAMILIES
NOISES=('original','independent_gaussian','heavy_tails_t5','hidden_E_scale_jump')
SUPPORT_SIZES=(2,5,8,12)
PROBE_SEED=991240
ENERGY_PROTOCOL='per_location_standardized_L2_peak_bounded'
def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32: raise ValueError('A uint32 seed is required')
    return value
def _uint32_seed(namespace,base_seed):
    return int.from_bytes(hashlib.sha256(f'{namespace}|{base_seed}'.encode()).digest()[:4], 'big')
def _indices(indices, n, name):
    ix = np.asarray(indices)
    if (ix.ndim != 1 or ix.dtype.kind not in "iu" or not len(ix)
            or np.any(ix < 0) or np.any(ix >= n) or len(np.unique(ix)) != len(ix)):
        raise ValueError(f"{name} must contain unique valid integer indices")
    return ix.astype(np.int64, copy=True)

def make_preprocessing(d, train_indices):
    """Refit B2 and all condition statistics on exactly train_indices subset G.

    Reads only observed_raw, groups, terrain_raw, xy and G.  Query chemistry is
    transformed to return r, but is never inspected for fitted statistics.
    Consequently query NaNs are allowed in r, while train chemistry must be
    finite.  All F rows are deterministic transforms of observed conditions.
    """
    observed = np.asarray(d["observed_raw"], dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 12:
        raise ValueError("observed_raw must be N-by-12")
    n = len(observed)
    train = _indices(train_indices, n, "train_indices")
    g = _indices(d["G"], n, "G")
    if not np.isin(train, g).all():
        raise ValueError("Preprocessing fit indices must be a subset of G")
    xy = np.asarray(d["xy"], dtype=float)
    if xy.shape != (n, 2) or not np.isfinite(xy).all():
        raise ValueError("Finite aligned coordinates are required")
    b2 = old.fit_b2(observed, d["groups"], train)
    terrain, terrain_mean, terrain_scale = old.standardize_terrain(d["terrain_raw"], train)
    if terrain.shape != (n, 3) or not np.isfinite(terrain).all():
        raise ValueError("Finite aligned terrain conditions are required")
    c = np.column_stack((terrain, b2["support_n"] / (b2["support_n"] + 50.),
                         b2["fallback"].astype(float)))
    center = xy[train].mean(0)
    cmean = c[train].mean(0)
    cscale = np.maximum(c[train].std(0), .1)
    spatial = xy - center
    columns = [(c - cmean) / cscale, spatial / 60000.]
    for wavelength in (30000., 60000., 120000.):
        phase = 2 * np.pi * spatial / wavelength
        columns.extend((np.sin(phase), np.cos(phase)))
    features = np.column_stack(columns).astype(np.float32)
    if features.shape != (n, 19) or not np.isfinite(features).all():
        raise ValueError("Invalid condition features")
    return dict(b2, F=features, features=features, c=c,
                r=(observed - b2["mu"]) / b2["sigma"], fit_indices=train,
                terrain_mean=terrain_mean, terrain_scale=terrain_scale,
                feature_center=center, feature_cmean=cmean, feature_cscale=cscale,
                terrain_train_mean=terrain_mean, terrain_train_scale=terrain_scale,
                B2_global_mu=b2["global_mu"], B2_global_sigma=b2["global_sigma"])

def inner_splits(d, n_splits=3, buffer_m=4000):
    """G-only contiguous y bands with exact Euclidean train/valid buffering.

    Stable ties use x then unique point ID.  Each G point is valid exactly once;
    some other G points are excluded from a fold by the buffer.  The coordinate
    system is the generator's local Cartesian metres, not an inferred EPSG.
    """
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("n_splits must be an integer >=2")
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m must be finite and nonnegative")
    xy = np.asarray(d["xy"], dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected N-by-2 coordinates")
    g = _indices(d["G"], len(xy), "G")
    ids = np.asarray(d["ids"])
    if (ids.shape != (len(xy),) or len(np.unique(ids)) != len(ids)
            or not np.isfinite(xy[g]).all() or n_splits > len(g) // 2):
        raise ValueError("Invalid split geometry or too many folds")
    order = np.lexsort((ids[g], xy[g, 0], xy[g, 1]))
    folds = []
    for fold, valid in enumerate(np.array_split(g[order], n_splits)):
        valid = np.sort(valid)
        candidates = g[~np.isin(g, valid)]
        distance2 = ((xy[candidates, None] - xy[None, valid])**2).sum(2)
        min_distances = np.sqrt(distance2.min(1))
        keep = min_distances >= float(buffer_m)
        train = np.sort(candidates[keep])
        if len(train) < 24 or len(valid) < 2:
            raise ValueError("Spatial buffer leaves insufficient training/valid support")
        folds.append(dict(fold=fold, train=train, valid=valid,
                          excluded_buffer=np.sort(candidates[~keep]),
                          buffer_m=float(buffer_m),
                          min_train_valid_distance_m=float(min_distances[keep].min()),
                          geometry="G-only equal-count contiguous y bands; Cartesian metres"))
    return folds
def _maximin_centers(xy, ids, seed, n_centers):
    """Coordinate median seed point, then farthest-point coverage; no chemistry."""
    if type(n_centers) is not int or not 1 <= n_centers <= len(xy):
        raise ValueError("Invalid maximin center count")
    _seed(seed)
    if len(np.unique(xy, axis=0)) < n_centers:
        raise ValueError("Too few distinct coordinates for requested centers")
    # Priority is assigned in ID order, preserving behavior under input row order.
    rng = np.random.default_rng(seed)
    priority = np.empty(len(xy), dtype=int)
    priority[np.argsort(ids, kind="stable")] = rng.permutation(len(xy))
    median_distance = ((xy-np.median(xy, axis=0))**2).sum(1)
    first = int(np.lexsort((priority, median_distance))[0])
    selected = [first]
    distance_to_selected = ((xy-xy[first])**2).sum(1)
    for _ in range(1, n_centers):
        eligible = distance_to_selected.copy()
        eligible[selected] = -np.inf
        nxt = int(np.lexsort((priority, -eligible))[0])
        selected.append(nxt)
        distance_to_selected = np.minimum(distance_to_selected, ((xy-xy[nxt])**2).sum(1))
    return selected

def _directions(seed):
    rng = np.random.default_rng(_seed(seed))
    result = []
    for support in SUPPORT_SIZES:
        chosen = np.sort(rng.choice(12, size=support, replace=False))
        direction = np.zeros(12)
        direction[chosen] = rng.uniform(.5, 1.5, support)
        direction /= np.linalg.norm(direction)
        result.append((support, direction))
    return result
def generate_dataset(family,base_seed,noise):
    if family not in FAMILIES or noise not in NOISES: raise ValueError('Unsupported design')
    d = old.generate_synthetic(family, base_seed, 0.)
    if noise == 'heavy_tails_t5':
        d = helper._recompute_from_noise(d, helper._noise_fields(d['xy'], base_seed, noise), 0., legacy=old)
    elif noise == 'independent_gaussian':
        # Fixed independent-noise random stream and draw law.
        seed = _uint32_seed('M32_noise_v1|'+noise, base_seed)
        rng = np.random.Generator(np.random.PCG64(seed))
        field = rng.standard_normal((len(d['ids']), 12))
        d = helper._recompute_from_noise(d, dict(noise_raw=.45*field,
            noise_element_field=field, noise_seed=np.array(seed, dtype=np.uint32)), 0., legacy=old)
    elif noise == 'hidden_E_scale_jump':
        # Preserve old generated outer arrays, but do not use outer E for fitting
        # or evaluation. The designated inner-valid stress is constructed below.
        scale = np.ones(len(d['ids'])); scale[d['E']] = 1.5
        d = helper._recompute_from_noise(d, {'noise_raw': d['noise_raw']*scale[:, None]}, 0., legacy=old)
    d.update(noise_mode=np.array(noise))
    if not np.allclose(d['observed_raw'], d['bg_raw']+d['a_train_raw']+d['noise_raw'], atol=1e-12):
        raise RuntimeError('Raw additivity')
    return d

def split(d, fold=0, buffer_m=4000):
    outer = inner_splits(d, n_splits=3, buffer_m=buffer_m)[fold]
    train, heldout = outer['train'].copy(), outer['valid'].copy()
    ids = np.asarray(d['ids'])
    ranked = sorted(heldout.tolist(), key=lambda i: hashlib.sha256(
        ('M34_inner_calibration_v1|'+str(ids[i])).encode('utf-8')).hexdigest())
    if len(ranked) != 280: raise ValueError('Expected 280 heldout G rows')
    cal, valid = np.sort(ranked[:140]), np.sort(ranked[140:])
    hidden = str(d['noise_mode']) == 'hidden_E_scale_jump'
    query = np.asarray(d['observed_raw'], float).copy()
    if hidden: query[valid] += .5*np.asarray(d['noise_raw'])[valid]
    pp = make_preprocessing(dict(d, observed_raw=query), train)
    return dict(train=train,cal=cal,valid=valid,exclude=outer['excluded_buffer'],
        pp=pp,query_observed_raw=query,audit=dict(
            source_role='original_G_only',fold=fold,buffer_m=float(buffer_m),
            train_n=len(train),calibration_n=len(cal),validation_n=len(valid),
            min_train_heldout_distance_m=outer['min_train_valid_distance_m'],
            calibration_validation_spatially_independent=False,
            calibration_validation_split='ID-SHA256 halves within one held-out spatial band',
            outer_C_E_used=False,hidden_validation_scale_jump=hidden,
            hidden_validation_modified_ids=ids[valid].tolist() if hidden else [],
            scale=1.5 if hidden else 1.,coordinate_units='local Cartesian metres',
            statistical_unit='independent base realization; folds/initializations/probes nested'))

def probes(d, indices, train, seed=PROBE_SEED, n_centers=6):
    """Positive sparse probes with bounded standardized L2 amplitude."""
    indices, train = np.asarray(indices), np.asarray(train)
    if not np.isin(indices,d['G']).all() or not np.isin(train,d['G']).all() or np.intersect1d(indices,train).size:
        raise ValueError('Disjoint G-only query and train required')
    xy = d['xy'][indices]; ids=d['ids'][indices]
    scale=old.robust_scale(d['observed_raw'][train])
    centers=_maximin_centers(xy,ids,seed,n_centers)
    directions=_directions(seed)
    theta=np.deg2rad(20.);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    out=[]
    for ci,local_index in enumerate(centers):
        delta=xy-xy[local_index]; local=delta@rotation
        qmap={'compact':(delta**2).sum(1)/6000.**2,
              'broad':(delta**2).sum(1)/16000.**2,
              'elongated':(local[:,0]/5000.)**2+(local[:,1]/19000.)**2}
        for shape,q in qmap.items():
            weights=np.where(q<1.,(1.-q)**2,0.);receiver=weights>0
            if not receiver.any() or receiver.all(): raise ValueError('Fixed probe lacks controls/receivers')
            for support,direction in directions:
                for amplitude in (.25,.5):
                    delta_raw=amplitude*weights[:,None]*direction[None]*scale
                    out.append(dict(case_id=f'c{ci:02d}_{shape}_s{support:02d}_a{amplitude:g}',
                        center_index=ci,center_local_index=int(local_index),center_id=str(ids[local_index]),
                        center_xy_m=xy[local_index].copy(),shape=shape,support_size=support,
                        amplitude=amplitude,direction=direction.copy(),weights=weights.copy(),
                        receiver=receiver.copy(),receiver_count=int(receiver.sum()),delta_raw=delta_raw,
                        training_raw_scale=scale.copy(),energy_protocol=ENERGY_PROTOCOL))
    if len(out)!=144:raise RuntimeError('Incomplete probe grid')
    return out

def construct_input_arrays(d,fold,buffer_m):
    """Pure construction also used by known-base fixtures; no filesystem writes."""
    sp=split(d,fold,buffer_m);pp=sp['pp'];tr,ca,va=sp['train'],sp['cal'],sp['valid'];query=sp['query_observed_raw']
    dose_train=tr if buffer_m==4000 else split(d,fold,4000)['train']
    cases=probes(d,va,dose_train);mu=pp['mu'][va];sigma=pp['sigma'][va]
    raw_deltas=np.stack([c['delta_raw'] for c in cases])
    sp['audit'].update(dose_reference_buffer_m=4000,dose_training_indices=dose_train.tolist(),
        buffer_sensitivity_common_physical_A=True,dose_statistics_used_in_network_preprocessing=False)
    a=dict(train_x=pp['r'][tr].astype(np.float32),train_c=pp['F'][tr].astype(np.float32),
        train_ids=d['ids'][tr],train_xy=d['xy'][tr],train_mu=pp['mu'][tr],train_sigma=pp['sigma'][tr],
        cal_x=pp['r'][ca],cal_c=pp['F'][ca].astype(np.float32),cal_ids=d['ids'][ca],cal_xy=d['xy'][ca],
        cal_mu=pp['mu'][ca],cal_sigma=pp['sigma'][ca],bg_raw_cal=d['bg_raw'][ca],observed_raw_cal=query[ca],
        val_x=(query[va]-mu)/sigma,val_c=pp['F'][va].astype(np.float32),val_ids=d['ids'][va],
        val_xy=d['xy'][va],query_xy=d['xy'][va],mu=mu,sigma=sigma,
        bg=(d['bg_raw'][va]-mu)/sigma,bg_raw_valid=d['bg_raw'][va],observed_raw_valid=query[va],
        deltas=raw_deltas/sigma,deltas_raw=raw_deltas,receivers=np.stack([c['receiver'] for c in cases]),
        train=tr,cal=ca,valid=va,exclude=sp['exclude'],original_G=d['G'],original_C=d['C'],original_E=d['E'],
        train_raw_scale=cases[0]['training_raw_scale'],dose_training_indices=dose_train,elements=d['elements'],
        noise_raw_valid=query[va]-d['bg_raw'][va],terrain_mean=pp['terrain_mean'],terrain_scale=pp['terrain_scale'],
        feature_center=pp['feature_center'],feature_cmean=pp['feature_cmean'],feature_cscale=pp['feature_cscale'],
        B2_global_mu=pp['global_mu'],B2_global_sigma=pp['global_sigma'])
    metadata=[{k:v for k,v in c.items() if k not in ('delta_raw','receiver')} for c in cases]
    return a,dict(audit=sp['audit'],cases=metadata,n_cases=144,
        schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items()})
