"""Noise laws and reconstruction of response-derived quantities."""
import hashlib
import numpy as np
from . import analytic_fields
NOISE_MODES=('original','heavy_tails_t5','mixed_sign_element_correlation','spatially_correlated_errors')
GROUPS=analytic_fields.GROUP_CHANNELS
def _noise_seed(seed, mode):
    raw = f'M13_noise_v1|{seed}|{mode}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & ((1 << 63) - 1)

def _noise_fields(xy, seed, mode):
    """Small-fixture-testable noise formula; no dataset/role access or writes."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Noise coordinates must be finite Nx2.')
    if mode not in NOISE_MODES[1:]:
        raise ValueError('This helper only constructs the declared stress modes.')
    n = len(xy)
    actual_seed = _noise_seed(seed, mode)
    rng = np.random.Generator(np.random.PCG64(actual_seed))
    common = rng.standard_normal((n, 1))
    individual = rng.standard_normal((n, 12))
    shared_field, individual_field = common.copy(), individual.copy()
    row_scale = np.ones((n, 1), dtype=np.float64)
    norm = np.ones(n, dtype=np.float64)
    signs = np.ones(12, dtype=np.float64)
    if mode == 'heavy_tails_t5':
        row_scale = np.sqrt(3. / rng.chisquare(5., size=(n, 1)))
    elif mode == 'mixed_sign_element_correlation':
        signs = np.array([1., -1.] * 6)
    elif mode == 'spatially_correlated_errors':
        fields = np.column_stack((common, individual))
        smooth = np.empty_like(fields)
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            distance2 = ((xy[start:stop, None, :] - xy[None, :, :])**2).sum(2)
            kernel = np.exp(-distance2 / (2. * 4000.**2))
            kernel[distance2 > 12000.**2] = 0.
            norm[start:stop] = np.sqrt((kernel * kernel).sum(1))
            smooth[start:stop] = (kernel @ fields) / norm[start:stop, None]
        shared_field, individual_field = smooth[:, :1], smooth[:, 1:]
    noise = .45 * (np.sqrt(.35) * shared_field * signs[None, :]
                   + np.sqrt(.65) * individual_field) * row_scale
    return {'noise_raw': noise, 'common_normal': common, 'element_normal': individual,
            'noise_shared_field': shared_field, 'noise_element_field': individual_field,
            'noise_row_scale': row_scale, 'noise_spatial_row_norm': norm,
            'noise_factor_signs': signs, 'noise_seed': np.array(actual_seed, dtype=np.int64)}

def _recompute_from_noise(base, fields, contamination, legacy=None):
    """Rebuild every response-derived array, including B2, after replacing noise."""
    old = analytic_fields if legacy is None else legacy
    d = {key: np.array(value, copy=True) for key, value in base.items()}
    n = len(d['ids']); g = d['G']
    noise = np.asarray(fields['noise_raw'], dtype=np.float64)
    if noise.shape != (n, 12) or not np.isfinite(noise).all():
        raise ValueError('Invalid replacement noise.')
    clean = d['bg_raw'] + noise
    dose = old.robust_scale(clean[g])
    candidates = d['contamination_candidate_indices']
    count = int(round(.05 * len(g)))
    if len(candidates) < count or not np.isin(candidates, g).all():
        raise ValueError('Invalid train-only contamination candidates.')
    added = np.zeros((n, 12), dtype=np.float64)
    selected = np.zeros(n, dtype=bool)
    if contamination == .05:
        picked = candidates[:count]
        selected[picked] = True
        added[np.ix_(picked, GROUPS['G1'])] = 2. * dose[list(GROUPS['G1'])]
    observed = clean + added
    b2 = old.fit_b2(observed, d['groups'], g)
    terrain, terrain_mean, terrain_scale = old.standardize_terrain(d['terrain_raw'], g)
    mu, sigma = b2['mu'], b2['sigma']
    d.update(fields)
    d.update(clean_raw=clean, observed_raw=observed, a_train_raw=added,
             contamination_selected=selected, contamination=np.array(contamination),
             dose_scale_raw=dose, mu=mu, sigma=sigma, r=(observed-mu)/sigma,
             observed_r=(observed-mu)/sigma, bg_true_r=(d['bg_raw']-mu)/sigma,
             noise_true_r=noise/sigma, support_n=b2['support_n'], fallback=b2['fallback'],
             B2_global_mu=b2['global_mu'], B2_global_sigma=b2['global_sigma'],
             c=np.column_stack((terrain, b2['support_n']/(b2['support_n']+50.),
                                b2['fallback'].astype(np.float64))),
             terrain_train_mean=terrain_mean, terrain_train_scale=terrain_scale)
    return d
