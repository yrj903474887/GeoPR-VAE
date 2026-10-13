"""Coordinate-only spatial partition helpers. Units: EPSG:32649 metres."""
import numpy as np
from scipy.spatial import cKDTree

def coordinate_partition(xy, n_folds=5, sample_ids=None):
    """Deterministic balanced rectangular partition using only projected coordinates.

    Returns integer folds 1..k and a nested decision tree. No response, category,
    mineral locality or score enters partition construction. An unsplittable
    identical-coordinate boundary raises instead of silently splitting a place.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < n_folds:
        raise ValueError('Expected at least k finite 2D points')
    if not np.isfinite(xy).all():
        raise ValueError('Coordinates must be finite')
    if sample_ids is None:
        sample_ids = np.arange(len(xy)).astype(str)
    sample_ids = np.asarray(sample_ids, dtype=str)
    labels = np.zeros(len(xy), dtype=int)
    def recurse(idx, k, first):
        if k == 1:
            labels[idx] = first
            return {'fold': int(first), 'n': int(len(idx))}
        ranges = np.ptp(xy[idx], axis=0)
        axes = np.argsort(-ranges, kind='stable')
        left_k = k // 2
        ideal = int(round(len(idx) * left_k / k))
        chosen = None
        for axis in axes:
            order = np.lexsort((sample_ids[idx], xy[idx, 1-axis], xy[idx, axis]))
            sorted_idx = idx[order]
            gap_positions = np.flatnonzero(np.diff(xy[sorted_idx, axis]) > 0) + 1
            gap_positions = gap_positions[(gap_positions >= left_k) & (len(idx)-gap_positions >= k-left_k)]
            if len(gap_positions):
                split_n = int(gap_positions[np.argmin(np.abs(gap_positions - ideal))])
                chosen = (int(axis), sorted_idx, split_n)
                break
        if chosen is None:
            raise ValueError('Cannot divide identical coordinates into contiguous folds')
        axis, sorted_idx, split_n = chosen
        threshold = float((xy[sorted_idx[split_n-1], axis] + xy[sorted_idx[split_n], axis]) / 2)
        return {'axis': axis, 'axis_name': ['EASTING_UTM49N','NORTHING_UTM49N'][axis],
                'threshold_m': threshold, 'comparison': '<=', 'n': int(len(idx)),
                'left': recurse(sorted_idx[:split_n], left_k, first),
                'right': recurse(sorted_idx[split_n:], k-left_k, first+left_k)}
    tree = recurse(np.arange(len(xy)), int(n_folds), 1)
    assert np.array_equal(labels, assign_fold(xy, tree))
    return labels, tree

def assign_fold(xy, tree):
    """Apply frozen coordinate decision boundaries; does not check AOI membership."""
    xy = np.asarray(xy, dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Coordinates must be finite')
    labels = np.zeros(len(xy), dtype=int)
    def walk(idx, node):
        if 'fold' in node:
            labels[idx] = node['fold']
        else:
            m = xy[idx, node['axis']] <= node['threshold_m']
            walk(idx[m], node['left'])
            walk(idx[~m], node['right'])
    walk(np.arange(len(xy)), tree)
    return labels

def buffered_train_indices(xy, test_mask, buffer_m):
    """Retain potential training centers strictly farther than buffer from test centers."""
    xy = np.asarray(xy, dtype=float)
    test_mask = np.asarray(test_mask, dtype=bool)
    train_idx = np.flatnonzero(~test_mask)
    d, _ = cKDTree(xy[test_mask]).query(xy[train_idx], k=1)
    return train_idx[d > buffer_m], train_idx[d <= buffer_m], d
