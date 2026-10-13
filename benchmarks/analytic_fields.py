"""Analytic controlled fields; no regional measurements are distributed."""
from __future__ import annotations
import numpy as np
FAMILIES = ('smooth_nonlinear', 'background_step', 'omitted_spatial_factor')
ELEMENTS = np.array(['Au','Ag','As','Bi','Cu','F','Mo','Pb','Sb','Sn','W','Zn'])
GROUP_CHANNELS = {'G1': (0,1,2,8), 'G2': (4,7,11), 'G3': (10,9,6,3), 'G4': (5,)}
EPS=1e-12
def _matrix(values, width, name, finite=True):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or len(result) == 0:
        raise ValueError(f"{name} must be nonempty [N,{width}]")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {name}")
    return result

def _indices(indices, n):
    ix = np.asarray(indices)
    if ix.ndim != 1 or ix.dtype.kind not in "iu" or not len(ix):
        raise ValueError("G must be nonempty integer indices")
    if np.any(ix < 0) or np.any(ix >= n) or len(np.unique(ix)) != len(ix):
        raise ValueError("Invalid or repeated G index")
    return ix.astype(np.int64)

def robust_scale(train_values):
    """MAD -> linear IQR/1.349 -> sample SD; fail if no positive scale exists."""
    y = _matrix(train_values, 12, "training observations")
    if len(y) < 2:
        raise ValueError("At least two training observations required")
    location = np.median(y, axis=0)
    scale = 1.4826*np.median(np.abs(y-location), axis=0)
    iqr = np.diff(np.quantile(y, [.25,.75], axis=0, method="linear"), axis=0)[0]/1.349
    scale = np.where(scale > EPS, scale, iqr)
    sd = y.std(axis=0, ddof=1)
    scale = np.where(scale > EPS, scale, sd)
    if not np.isfinite(scale).all() or np.any(scale <= EPS):
        raise ValueError("Training global scale is unavailable")
    return scale

def fit_b2(observed_raw, groups, G):
    """Return mu/sigma/support/fallback, using only G response values to fit.

    The caller may pass all groups to address output references. Evaluation
    responses never enter a fitted statistic, including validation of scales.
    """
    y = _matrix(observed_raw, 12, "observations", finite=False)
    groups = np.asarray(groups)
    if groups.shape != (len(y),) or groups.dtype.kind not in "iu":
        raise ValueError("groups must be a row-aligned integer vector")
    G = _indices(G, len(y))
    train = y[G]
    if not np.isfinite(train).all():
        raise ValueError("Nonfinite training observations")
    global_mu = np.median(train, axis=0)
    global_sigma = robust_scale(train)
    mu = np.broadcast_to(global_mu, y.shape).copy()
    sigma = np.broadcast_to(global_sigma, y.shape).copy()
    support = np.zeros(len(y), dtype=np.int64)
    fallback = np.ones(len(y), dtype=bool)
    for group in np.unique(groups[G]):
        train_group = train[groups[G] == group]
        target = groups == group
        count = len(train_group)
        support[target] = count
        if count < 5:
            continue
        med = np.median(train_group, axis=0)
        mad = 1.4826*np.median(np.abs(train_group-med), axis=0)
        alpha = count/(count+50.)
        mu[target] = alpha*med + (1.-alpha)*global_mu
        sigma[target] = np.maximum(np.sqrt(alpha*mad**2+(1.-alpha)*global_sigma**2),
                                   .1*global_sigma)
        fallback[target] = False
    return {"mu":mu, "sigma":sigma, "support_n":support, "fallback":fallback,
            "global_mu":global_mu, "global_sigma":global_sigma}

def standardize_terrain(terrain_raw, G):
    t = _matrix(terrain_raw, 3, "terrain", finite=False)
    G = _indices(G, len(t))
    if not np.isfinite(t[G]).all():
        raise ValueError("Nonfinite training terrain")
    mean, scale = t[G].mean(0), t[G].std(0, ddof=0)
    if np.any(scale <= EPS) or not np.isfinite(scale).all():
        raise ValueError("Unavailable training terrain scale")
    return (t-mean)/scale, mean, scale

def training_graph(xy, G):
    """G-local i<j pairs; radius=3 median 1NN, Gaussian distance weights."""
    xy = _matrix(xy, 2, "coordinates", finite=False)
    G = _indices(G, len(xy))
    train_xy = xy[G]
    if len(G) < 2 or not np.isfinite(train_xy).all():
        raise ValueError("At least two finite training coordinates required")
    delta = train_xy[:,None,:]-train_xy[None,:,:]
    d2 = np.einsum("ijk,ijk->ij",delta,delta)
    np.fill_diagonal(d2,np.inf)
    length = float(np.median(np.sqrt(d2.min(1))))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Nonpositive training nearest-neighbour length")
    radius = 3.*length
    i,j = np.where(np.triu(d2 < radius**2, k=1))
    edges = np.column_stack((i,j)).astype(np.int64)
    weights = np.exp(-.5*d2[i,j]/length**2)
    if len(edges) == 0 or not np.isfinite(weights).all():
        raise ValueError("Empty or invalid training graph")
    return edges, weights, np.array(length), np.array(radius)

def background_formula(family, terrain_raw, normalized_xy, spatial_coefficients):
    """Fixed analytic family definitions. No neural network generates truth."""
    if family not in FAMILIES:
        raise ValueError("Unknown synthetic family")
    t = _matrix(terrain_raw,3,"terrain")
    uv = _matrix(normalized_xy,2,"normalized coordinates")
    if len(t) != len(uv):
        raise ValueError("Row mismatch")
    j = np.arange(12, dtype=np.float64)
    coefficients = np.column_stack((.55*np.cos(.45*j),
                                   .4*np.sin(.35*j+.3),
                                   .35*np.cos(.6*j+.2),
                                   .25*np.sin(.5*j+.7)))
    features = np.column_stack((np.sin(np.pi*t[:,0]),
                                (t[:,1]-.5)**2,
                                np.tanh(2.*t[:,2]),
                                t[:,0]*t[:,1]))
    smooth = (1.+.1*j)[None,:] + features@coefficients.T
    step_coefficients = .75*np.cos(.37*j+.15)
    step = (uv[:,0] >= .6).astype(np.float64)
    ax,bx = np.meshgrid(np.linspace(0.,1.,5),np.linspace(0.,1.,4))
    centers = np.column_stack((ax.ravel(),bx.ravel()))
    coef = np.asarray(spatial_coefficients,dtype=np.float64)
    if coef.shape != (20,) or not np.isfinite(coef).all():
        raise ValueError("Omitted-field coefficients must be finite [20]")
    d2 = ((uv[:,None,:]-centers[None,:,:])**2).sum(2)
    kernel = np.exp(-d2/(2.*.12**2))
    omitted = (kernel@coef)/np.sqrt((kernel**2).sum(1))
    omitted_coefficients = .6*np.sin(.4*j+.5)
    bg = smooth.copy()
    if family == "background_step":
        bg += step[:,None]*step_coefficients[None,:]
    elif family == "omitted_spatial_factor":
        bg += omitted[:,None]*omitted_coefficients[None,:]
    return bg, coefficients, step, omitted

def generate_synthetic(family, seed, contamination=0.):
    if family not in FAMILIES:
        raise ValueError("Unknown synthetic family")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if isinstance(contamination,bool) or contamination not in (0.,.05):
        raise ValueError("Only preregistered contamination=0 or .05 is allowed")
    # Fixed draw order shared across all families and contamination states.
    rng = np.random.Generator(np.random.PCG64(seed))
    col = np.tile(np.arange(40,dtype=np.int64),30)
    row = np.repeat(np.arange(30,dtype=np.int64),40)
    xy = 2000.*np.column_stack((col,row)) + rng.uniform(-100.,100.,size=(1200,2))
    uv = xy/np.array([78000.,58000.])
    u,v = uv.T
    terrain_noise = rng.standard_normal((1200,3))
    terrain_raw = np.column_stack((u+.2*np.sin(2*np.pi*v),
                                   v+.2*np.cos(2*np.pi*u),
                                   np.sin(np.pi*(u+v))+.25*np.cos(3*np.pi*v))) + .04*terrain_noise
    spatial_coefficients = rng.standard_normal(20)
    common_normal = rng.standard_normal((1200,1))
    element_normal = rng.standard_normal((1200,12))
    noise_raw = .45*(np.sqrt(.35)*common_normal+np.sqrt(.65)*element_normal)
    role = np.full(1200,"buffer",dtype="U6")
    role[(col <= 19)|(col >= 32)] = "G"
    role[(col >= 22)&(col <= 25)] = "C"
    role[(col >= 27)&(col <= 30)] = "E"
    G,C,E = [np.flatnonzero(role==name).astype(np.int64) for name in ("G","C","E")]
    groups = (row*4//30).astype(np.int64)
    bg_raw, bg_coefficients, step, omitted = background_formula(family,terrain_raw,uv,spatial_coefficients)
    clean_raw = bg_raw + noise_raw
    dose_scale_raw = robust_scale(clean_raw[G])
    contamination_candidates = rng.permutation(G)[:42]
    a_train_raw = np.zeros((1200,12),dtype=np.float64)
    contamination_selected = np.zeros(1200,dtype=bool)
    if contamination == .05:
        contamination_selected[contamination_candidates] = True
        a_train_raw[np.ix_(contamination_candidates,GROUP_CHANNELS["G1"])] = 2.*dose_scale_raw[list(GROUP_CHANNELS["G1"])]
    observed_raw = clean_raw+a_train_raw
    b2 = fit_b2(observed_raw,groups,G)
    standardized,terrain_mean,terrain_scale = standardize_terrain(terrain_raw,G)
    c = np.column_stack((standardized,b2["support_n"]/(b2["support_n"]+50.),b2["fallback"].astype(np.float64)))
    edges,distance_weights,length,radius = training_graph(xy,G)
    mu,sigma = b2["mu"],b2["sigma"]
    result = {
        "ids":np.array([f"S{i:06d}" for i in range(1200)],dtype="U7"),
        "r":(observed_raw-mu)/sigma, "c":c, "xy":xy, "G":G,"C":C,"E":E,
        "edges":edges,"distance_weights":distance_weights,"active":np.ones(1200,dtype=bool),
        "quality_valid":np.ones(1200,dtype=bool),"bg_true_r":(bg_raw-mu)/sigma,
        "observed_raw":observed_raw,"bg_raw":bg_raw,"sigma":sigma,"mu":mu,
        "a_train_raw":a_train_raw,"role":role,"family":np.array(family),
        "seed":np.array(seed,dtype=np.int64),"contamination":np.array(contamination,dtype=np.float64),
        "dose_scale_raw":dose_scale_raw,"elements":ELEMENTS.copy(),"groups":groups,
        "noise_raw":noise_raw,"noise_true_r":noise_raw/sigma,
        "common_normal":common_normal,"element_normal":element_normal,
        "terrain_raw":terrain_raw,"terrain_noise":terrain_noise,
        "terrain_train_mean":terrain_mean,"terrain_train_scale":terrain_scale,
        "support_n":b2["support_n"],"fallback":b2["fallback"],
        "B2_global_mu":b2["global_mu"],"B2_global_sigma":b2["global_sigma"],
        "graph_median_1nn_m":length,"graph_radius_m":radius,
        "contamination_selected":contamination_selected,
        "contamination_candidate_indices":contamination_candidates,
        "row":row,"col":col,"normalized_xy":uv,
        "background_coefficients":bg_coefficients,"step_factor":step,
        "omitted_factor":omitted,"spatial_coefficients":spatial_coefficients,
        "schema_version":np.array("analytic_synthetic_v1")}
    if any(not isinstance(v,np.ndarray) or v.dtype.hasobject for v in result.values()):
        raise RuntimeError("Unsafe synthetic serialization schema")
    return result
