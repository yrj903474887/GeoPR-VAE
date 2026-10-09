"""IDW onto a caller-supplied valid grid; no automatic study-area masking."""
import numpy as np
from scipy.spatial import cKDTree
from .preprocessing import ELEMENTS


def midrank(values):
    ordered = np.sort(values)
    return (np.searchsorted(ordered, values, "left") + np.searchsorted(ordered, values, "right")) / (2 * len(values))


def grid_scores(xy, affine_scores, grid_xy):
    xy, scores, grid_xy = map(lambda a: np.asarray(a, float), (xy, affine_scores, grid_xy))
    if xy.ndim != 2 or xy.shape[1] != 2 or scores.shape != (len(xy), 13):
        raise ValueError("Expected xy[N,2], scores[N,13]")
    if grid_xy.ndim != 2 or grid_xy.shape[1] != 2 or not len(grid_xy) or not len(xy):
        raise ValueError("Expected nonempty grid_xy[M,2]")
    if not all(np.isfinite(a).all() for a in (xy, scores, grid_xy)):
        raise ValueError("Finite geometry and scores required")
    k = min(12, len(xy))
    distance, indices = cKDTree(xy).query(grid_xy, k=k)
    if k == 1:
        distance, indices = distance[:, None], indices[:, None]
    weights = 1 / np.maximum(distance, 1e-9) ** 2
    weights /= weights.sum(1, keepdims=True)
    neighbors = scores[indices]
    anchor = neighbors[:, :1]
    values = np.round(anchor[:, 0] + np.sum(weights[..., None] * (neighbors - anchor), axis=1), 12)
    percentiles = np.column_stack([midrank(values[:, j]) for j in range(1, 13)])
    groups = [["Cu", "Pb", "Zn", "Ag"], ["W", "Sn", "Mo", "Bi"]]
    association = np.column_stack([percentiles[:, [ELEMENTS.index(e) for e in group]].mean(1) for group in groups])
    return dict(xy=grid_xy, affine_scores=values, element_percentiles=percentiles,
                composite_percentile=midrank(values[:, 0]), association_percentiles=association,
                association_names=np.asarray(["Cu-Pb-Zn-Ag", "W-Sn-Mo-Bi"]))
