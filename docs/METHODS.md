# Model configuration

The command-line configuration specifies the model, preprocessing, calibration, and regional mapping settings.

| Modules | Role |
| --- | --- |
| `reference_base.py`, `reference.py` | Conditional reference networks, masked likelihood and posterior draws |
| `spatial.py`, `pseudo_training.py` | Spatial contrasts, recovery features and paired training fields |
| `recovery.py`, `training.py` | Signed nuisance estimation and preservation-guided optimization |
| `scoring_base.py`, `scoring.py` | Calibration, field normalization, spatial smoothing and anomaly scoring |
| `preprocessing_base.py`, `preprocessing.py` | Training-fitted group preprocessing, model conditions and calibration splits |
| `pipeline.py`, `mapping.py` | Training, prediction, interpolation and fixed element associations |

Reference optimization: full batch, AdamW, learning rate 0.0005, weight decay 0.001, gradient clipping 5, 600 epochs. GeoPR-VAE uses a Gaussian decoder, beta 1, one masked target, conditional prior width 32, positive protection weight 100 and kappa 0.01. Protection support sizes are 2 or 4 and amplitudes 0.25 or 0.5. Posterior log variances are clamped to [-6,6]; Gaussian decoder log variances to [-4,4]. Sixteen draws are keyed by sample ID and target rather than target chemistry. See `reference_base.py` for the exact perturbation and RNG construction.

Recovery optimization: width 32, zero-initialized final nuisance layer, 300 updates, AdamW learning rate 0.001, weight decay 0.0001, clipping 5. Loss weights are null 1, sparse 0.001, preservation 32 and off-support 0.1. Training uses at most 512 randomly selected training nodes, 24 pseudo-null fields, a 6 km spectral graph and 3/9 km recovery neighborhoods. Positive intervention axis scales are log-uniform 2–18 km; peak training-scaled L2 amplitudes are log-uniform 0.04–0.9. The uncentered element Gram matrix and Laplacian quadratic form are preserved by the pseudo-null transformations, within floating-point tolerance. Such residuals are not measured clean noise and may contain enrichment or reference bias.

Anomaly scoring uses calibration-MAD for AE/VAE and negative-half-energy adaptation for GeoPR-VAE. The MAD floor is 0.1 standardized response units; negative-energy floor is 0.0001; field-energy ratios are clipped to [0.25,9] and mixed with weight 0.5 before taking the square root. Score smoothing includes self, while recovery graphs exclude self. No row-power correction is applied. Calibration and each query field keep separate graphs. The implementation retains support for the 0/3/6/10 km scoring grid through the low-level API; the configured pipeline fixes 10 km as in the paper.

Group preprocessing is fitted only on training rows. Global robust scale follows MAD, IQR/1.349, then sample SD. Groups with at least five training samples use `alpha = n/(n+50)` shrinkage of the median and squared MAD toward global estimates, with a scale floor of 0.1 times global scale. Smaller or unseen groups use the global estimates. In the recovery head, residual MAD instead falls back directly to sample SD and has an absolute 0.1 floor. These are different scale definitions and are intentionally retained.
