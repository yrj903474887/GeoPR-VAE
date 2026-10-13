# GeoPR-VAE

Research implementation accompanying **GeoPR-VAE: Spatial-Contrast Residual Deep Learning for Multielement Weak Geochemical Anomaly Identification**.

This repository contains the conditional reference model, spatial-contrast recovery, training-only preprocessing, calibration and anomaly scoring, AE/VAE comparators, and regional grid interpolation. The repository also provides controlled experiments, patch-based comparison models, paired weak-enrichment evaluation, mineral-association metrics, and figure-generation tools. Data, trained weights, and generated results are supplied or produced locally and are not included in the repository.

## Install and verify

Use Python 3.10 or newer in a virtual environment. Install a CPU or CUDA build of PyTorch appropriate for your system, then install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m examples.smoke --save-input outputs/synthetic.npz --report outputs/smoke_report.json
```

Run commands from the repository root. The implementation executes on CPU and uses one PyTorch thread, for repeatable CPU execution. The smoke example generates a small synthetic field, runs all three methods, checks checkpoint and batching consistency, and evaluates a positive paired intervention. It uses 8 reference epochs and 12 recovery updates for speed. Its measurements are execution checks, not estimates of the paper's performance. No downloaded data are needed.

Additional checks are available through `python -m benchmarks.test_smoke` and `python -m scripts.smoke_evaluation --output outputs/evaluation_smoke --plots`. See [validation](docs/VALIDATION.md) for their scope. Dependency ranges in `requirements.txt` are installation constraints, not a claim that every supported combination produces identical numerical results.

## Complete workflows

[Experiment and figure workflows](docs/WORKFLOWS.md) documents the full six-method controlled matrix, buffered five-fold validation, 144 paired weak interventions, additional comparison networks, global deployment, mineral-association evaluation, cartography and tables. It includes exact command-line examples and file schemas. The core model interface is summarized below.

## Input schema

Supply one NumPy `.npz` file. Arrays must have the same row order. The element order is fixed:

`Au, Ag, As, Bi, Cu, F, Mo, Pb, Sb, Sn, W, Zn`

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `log10` | N × 12, finite floats | Base-10 logarithms of strictly positive concentrations; not standardized values |
| `xy` | N × 2, finite floats | Easting and northing in one projected CRS, in **metres** |
| `terrain` | N × 3, finite floats | Elevation (m), slope (degrees), and terrain-position index (m) |
| `groups` | N, integers | Anonymous statistical group codes; not verified geological classes |
| `ids` | N, unique strings | Stable sample identifiers; preserve leading zeros and keep the same IDs in paired interventions |
| `train` | Optional integer vector | Zero-based training indices |
| `calibration` | Optional integer vector | Zero-based calibration indices, disjoint from `train` |

Provide both index vectors or neither. Training requires at least 24 rows and calibration at least 4. Very small calibration pools can produce an infinite empirical threshold; at least 37 calibration rows are needed for a finite 95% order-statistic threshold under the fixed split. This threshold is not a guaranteed false-positive rate.

In the paper, Au and Ag are reported in **ppb**, and the other ten elements in **ppm**. Take `np.log10` in those units and use the same convention for training and later queries. Coordinates are WGS84 / UTM zone 49N, EPSG:32649. Longitude/latitude in degrees are not acceptable for the spatial kernels. Other regions may use a suitable local projected metre CRS, consistently for all arrays and grids.

Terrain variables were derived from 30 m elevation grids. TPI is elevation minus its 2 km neighborhood mean. Before exporting inputs, apply the desired data-quality mask: the paper required valid point terrain and at least 80% valid terrain coverage within 2 km. This package rejects missing values; it does not silently impute measurements, extract terrain from rasters, or convert detection-limit codes.

For example, once the arrays have been assembled:

```python
import numpy as np
np.savez_compressed(
    "data/region.npz", log10=np.log10(concentrations), xy=xy_metres,
    terrain=terrain, groups=anonymous_group_codes, ids=sample_ids.astype(str),
    train=train_indices, calibration=calibration_indices,
)
```

Create the `data/` directory first. Exclude mineral labels, known clean signals and other evaluation truth from these training inputs.

## Train and score

```bash
python -m geopr_vae train --input data/region.npz --output outputs/region --method all
```

`--method` also accepts `GeoPR-VAE`, `AE` or `VAE`. Defaults are the manuscript settings: 600 reference epochs, 300 GeoPR-VAE head updates, seed 20260929 and 16 posterior draws per target. `--head-seed` sets the recovery-head seed separately and defaults to `--seed`. For spatial validation, use the distinct seeds written by `benchmarks.prepare_region` as described below. The CLI and `pipeline.fit` apply these manuscript configurations. Low-level `reference.make_model` has separate general defaults; pass `pipeline.CONFIGS[method]` explicitly when using that constructor directly. Use `--teacher-epochs` and `--head-epochs` only when intentionally changing the training schedule. Outputs record the chosen schedule.

If the two index vectors are absent, the regional deployment split allocates 20% to calibration using 10 km tiles, Hamilton allocation and fixed sample-ID hashes. It uses geometry and IDs, without chemical values or mineral labels. It is a deployment split and does **not** impose the buffered spatial exclusion used for the paper's independent validation.

Each method writes:

- `model.pt`: fitted preprocessing, reference weights, optional recovery weights and calibration carriers.
- `prediction.npz`: predictions and scores for **every input row**, including training rows.
- `training.json`: configuration summary, fitted sample IDs and optimization history.
- `scoring.json`: calibration parameters, field factors and affine transformations.

Outputs are placed in a new directory to avoid silently replacing a previous run. Generated arrays, checkpoints and local data are excluded by `.gitignore`.

For an independently supplied query field using a saved model:

```bash
python -m geopr_vae predict --model outputs/region/GeoPR-VAE/model.pt --input data/query.npz --output outputs/query_prediction.npz
```

Query files contain the five required arrays and use the original measurement units, group definitions and coordinate system. The stored training statistics are reused; unseen groups fall back to the training global estimates. Calibration measurements are already represented in the checkpoint and are not refitted.

Pass one **complete spatial query field** at a time. Recovery neighborhoods, negative-energy adaptation and score smoothing depend on field membership. Splitting a region into geographic batches changes the method. The reference network's internal prediction batches do not change field membership. The Gaussian scorer retains the original dense N × N geometry and has quadratic memory use; the supplied workflow targets regional sample sets of the paper's size.

## Model and score outputs

Training-fitted robust group shrinkage transforms log concentrations into standardized response units, called B2 in the implementation. Conditions contain three standardized terrain variables, two group-support indicators, two coordinates scaled by 60 km, and twelve sine/cosine coordinate features at 30, 60 and 120 km wavelengths. No geological-unit or mineral-occurrence label is a network input.

GeoPR-VAE masks one chemical target, fits a conditional Gaussian reference, and averages 16 reproducibly keyed posterior draws for each separately masked target. A signed recovery head uses 3 km and 9 km leave-self-out residual contrasts. It predicts nuisance for subtraction through a unit residual pathway: `C = (x - R) - nuisance`, `D = R + C`. Training uses 24 paired pseudo-null fields and independent positive interventions; analytical clean truth never enters the objective.

| Configuration | AE | VAE | GeoPR-VAE |
| --- | ---: | ---: | ---: |
| Hidden width / latent dimension | 96 / 4 | 64 / 4 | 64 / 8 |
| Reference parameters | 25,552 | 13,984 | 16,696 |
| Recovery-head parameters | 0 | 0 | 3,308 |
| Prior | None | Standard Gaussian | Conditional Gaussian |
| Recovery | Enhanced residual readout | Enhanced residual readout | Signed nuisance subtraction |
| Score carrier | Reference residual | Reference residual | Learned correction C |

AE/VAE recovery uses `D = R + 0.75*(x-R) + 0.25*max(x-R-0.5, 0)`, as in the configured manuscript comparison. Their anomaly scores still use the reference residual, not this recovery output.

Calibration is split by a fixed salted ID hash into scale-fitting and score-transformation pools. Recovery graphs for these pools are computed separately. All methods use calibration median/MAD scaling; GeoPR-VAE additionally uses negative-energy adaptation. Signed channels are smoothed with a 10 km Gaussian bandwidth and 30 km cutoff. The composite is the mean of the largest three positive smoothed channels. Each channel receives a positive affine transformation fitted to the second calibration pool.

`prediction.npz` includes:

- `reference`, `residual`, `carrier`, `recovered`: N × 12 arrays in standardized response units.
- `reference_log10`, `recovered_log10`: N × 12 arrays converted back to the input log10 concentration units.
- `raw_scores`, `affine_scores`: N × 13 arrays, ordered **Composite, then the twelve elements**. Single-element scores remain signed.
- `threshold`, `above_threshold`: empirical threshold of the raw composite and its strict exceedance flag. These are descriptive readouts without a nominal FPR guarantee.
- Variance arrays for VAE/GeoPR-VAE: latent-posterior variation of decoder locations and decoder observation variance at fixed fitted weights, in standardized units. They are not parameter epistemic uncertainty. AE has no probabilistic observation variance.

## Regional grid and association scores

Prepare a `.npz` grid with `xy[M,2]` containing only valid cell centers, in the same metre CRS. Then:

```bash
python -m geopr_vae grid --prediction outputs/region/GeoPR-VAE/prediction.npz --grid data/valid_grid.npz --output outputs/region_grid.npz
```

The command uses 12-neighbor inverse-distance weighting with power 2, then converts each element to its within-grid midrank percentile. Cu–Pb–Zn–Ag and W–Sn–Mo–Bi scores are equal-weight means of the four corresponding element percentiles, without ranking again. The composite gets its own percentile. These values are relative responses, not mineralization probabilities.

The caller supplies the land, convex-hull and sampling-support mask. The paper used 500 m cells, complete cell coverage by land and the valid-sample hull, nearest-sample distance at most 5 km, and at least three samples within 10 km. Its fixed 75,121-cell mask and mineral inventory are not distributed. Do not extrapolate into unsupported cells and interpret the resulting grid as the original study map.

## Paired-intervention API

`geopr_vae.experiments.paired_intervention` accepts a fitted model, an independent query field and a nonnegative log10-space intervention. It reruns both the frozen reference and recovery for the original and perturbed fields using the same IDs and posterior-draw keys. It reports directional preservation gain in training-standardized units, relative pair error and off-support RMS. Optional clean truth is used only for evaluator-side recovery RMSE.

```python
from geopr_vae.experiments import paired_intervention
report, original, perturbed = paired_intervention(bundle, query, delta_log10, clean_log10)
```

Use the same geometry, training/calibration/evaluation roles and interventions across methods. For spatial validation, supply explicit `train` and `calibration` indices, fit one model per fold/buffer, then call `predict` separately on that fold's held-out query rows. Construct exclusions before fitting any preprocessing; check the intended geographic separation between every role pair. Do not interpret the default all-row deployment prediction as out-of-fold validation.

