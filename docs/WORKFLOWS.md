# Experiment and evaluation workflows

Run all commands from the repository root, after installing `requirements.txt`. For a tested installation and resource measurements, see [ENVIRONMENT.md](ENVIRONMENT.md). Input arrays and measurement units follow the [user guide](USER_GUIDE.md#input-schema). Outputs are local generated files; none are bundled with the source.

## Complete controlled experiment matrix

The frozen configuration in [configs/paper_experiments.json](../configs/paper_experiments.json) defines AE, VAE, GeoPR-VAE, Patch VAE (`VAE_patch`), GANomaly and Skip-GANomaly (`Skip_GANomaly`). The confirmation matrix comprises ten independent bases, three background families, four noise conditions, two buffers and three spatial folds: **720 fitted tasks per method, 4,320 tasks for all six methods**. Each task evaluates 144 paired interventions. Controlled folds are indexed 0–2; the regional folds below are indexed 1–5.

```bash
python -m benchmarks.controlled --config configs/paper_experiments.json --phase confirmation --output outputs/controlled --workers 8
python -m benchmarks.aggregate --config configs/paper_experiments.json --phase confirmation --run outputs/controlled --output outputs/controlled_tables
```

The schedules are method-specific: 600 reference epochs for the core AE/VAE/GeoPR-VAE models, 300 recovery updates for GeoPR-VAE, and 100 epochs for the three patch comparators. They are not all 100-epoch networks. To run only the patch extension, add `--methods VAE_patch GANomaly Skip_GANomaly`; to compare all six methods, all six complete method matrices are needed. The runner executes the requested task list on each invocation; use a new output directory to avoid overwriting an earlier run. `--save-predictions` additionally stores prediction arrays locally.

Use `--limit 1 --smoke` in a **separate output directory** for a one-epoch, two-probe execution check. The aggregator rejects these smoke results and rejects incomplete or duplicate research matrices. `--phase development` selects the separate development seed set; do not combine it with confirmation results.

The aggregator emits `base_means.csv`, `noise_base_means.csv`, `amplitude_base_means.csv`, a long table and a summary. Each controlled observation is one independent base mean, with nested families, folds and cases averaged first.

## Regional spatial validation and 144 weak interventions

### Prepare spatial roles

Start from the five required point arrays described above, before removing rows solely because of terrain quality. An optional Boolean `valid[N]` marks those rows. The partitioner first constructs five coordinate-based folds on the complete coordinate/ID frame; it then applies buffers and the validity mask. `log10` and `terrain` must be finite for retained rows. All coordinates and IDs in the spatial frame must be valid.

```bash
python -m benchmarks.prepare_region --input data/region_full.npz --mode cv --fold 1 --buffer-km 4 --output outputs/cv_b4_fold1
```

Repeat for folds 1–5 and buffers 4/8 km. Each output directory contains:

| File | Contents |
| --- | --- |
| `core_train.npz` | Valid observations with explicit training/calibration indices for core methods |
| `core_query.npz` | Only the held-out field, in identical order to patch validation rows |
| `patch_input.npz` | Training-standardized `train_*`, `cal_*`, `val_*` arrays |
| `settings.json` | Role counts, partition specification and separate reference/head/patch seeds |

Patch arrays use `ROLE_x[N,12]`, `ROLE_xy[N,2]`, `ROLE_ids[N]` and `ROLE_c[N,K]`, where `ROLE` is `train`, `cal` or `val`. The preparation tool fits concentration and condition transforms on training rows only. Patch construction and calibration scoring respect the separate spatial roles.

Use the prepared seeds directly; for example, save and run this Python recipe from the repository root:

```python
import json
import subprocess
import sys
from pathlib import Path

run = Path("outputs/cv_b4_fold1")
seeds = json.loads((run / "settings.json").read_text())["seeds"]
subprocess.run([
    sys.executable, "-m", "geopr_vae", "train",
    "--input", str(run / "core_train.npz"), "--output", str(run / "core"),
    "--method", "all", "--seed", str(seeds["reference_seed"]),
    "--head-seed", str(seeds["head_seed"]),
], check=True)
```

The reference seed is `2026091202 + 104729*(fold-1)`; the head seed is `2026410101 + 104729*fold`; the patch seed is `2026101001 + 104729*(fold-1)`. These are recorded in [configs/regional_experiments.json](../configs/regional_experiments.json). Core training outputs predictions for all valid input rows; held-out evaluation instead uses `core_query.npz` and the saved model.

### Prepare, predict and evaluate paired fields

```bash
python -m benchmarks.weak prepare --input outputs/cv_b4_fold1/patch_input.npz --fold 1 --buffer-km 4 --output outputs/cv_b4_fold1/weak_input.npz
python -m benchmarks.weak core-predict --input outputs/cv_b4_fold1/weak_input.npz --query outputs/cv_b4_fold1/core_query.npz --model outputs/cv_b4_fold1/core/GeoPR-VAE/model.pt --output outputs/cv_b4_fold1/geopr_weak_predictions.npz
python -m benchmarks.weak evaluate --input outputs/cv_b4_fold1/weak_input.npz --predictions outputs/cv_b4_fold1/geopr_weak_predictions.npz --output outputs/cv_b4_fold1/geopr_weak.json
```

Repeat core inference/evaluation for AE and VAE. The same trained model predicts the original field and all 144 perturbed fields. The input IDs and posterior keys remain fixed. `after_*` metrics evaluate the perturbed score; `delta_*` metrics evaluate its change relative to the original field. The weak-response analyses use `delta_auc` and `delta_core_recall_top10`.

For each patch method, train **once on the prepared weak input**, which also performs all 145 predictions. Read its seed from `settings.json`, as in the core recipe:

```python
for method in ("VAE_patch", "GANomaly", "Skip_GANomaly"):
    subprocess.run([
        sys.executable, "-m", "benchmarks.patch_runner",
        "--input", str(run / "weak_input.npz"),
        "--output", str(run / method), "--method", method,
        "--seed", str(seeds["patch_seed"]), "--epochs", "100", "--save-model",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "benchmarks.weak", "evaluate",
        "--input", str(run / "weak_input.npz"),
        "--predictions", str(run / method / "predictions.npz"),
        "--output", str(run / (method + "_weak.json")),
    ], check=True)
```

After all folds/buffers/methods finish, assemble a CSV manifest with columns `method,buffer_km,fold,path`; `path` points to each evaluation JSON and may be relative to the manifest. The complete comparison requires all six methods and five fold means for each buffer. Aggregation verifies 144 distinct cases in every supplied JSON.

```bash
python -m scripts.aggregate_results weak-json --manifest outputs/weak_manifest.csv --output outputs/weak_tables
```

For already exported case-level CSVs, `scripts.aggregate_results controlled --cases FILE --output DIR` and `scripts.aggregate_results weak --cases FILE --output DIR` provide equivalent table interfaces; required columns are documented in `evaluation/controlled.py`.

## Additional comparison networks

`benchmarks.patch_runner` also accepts `VAE_GAN`, `VAE_WGAN`, `VAE_WGANGP`, `f_AnoGAN` and `Geo_Hgan`. These additional methods are separate from the six-method controlled configuration. The first seven patch variants are independent implementations of the VAE, adversarial, GANomaly, Skip-GANomaly and f-AnoGAN architectures with a shared geochemical patch interface. Original method attribution is retained in `benchmarks/models.py`; using this code does not make these independent implementations the authors' official releases.

Geo-Hgan imports the authors' network classes from a separately obtained checkout of [their official repository](https://github.com/cbn001012/Geo-Hgan-Unsupervised-anomaly-detection-in-geochemical-data-via-latent-space-learning). Obtain it outside this repository and retain its license and attribution:

```bash
git clone https://github.com/cbn001012/Geo-Hgan-Unsupervised-anomaly-detection-in-geochemical-data-via-latent-space-learning ../Geo-Hgan-author
git -C ../Geo-Hgan-author checkout c2cce554b4ff8571c2f315ac8e11308f45443a2c
python -m benchmarks.patch_runner --input outputs/cv_b4_fold1/weak_input.npz --output outputs/cv_b4_fold1/Geo_Hgan --method Geo_Hgan --seed 2026101001 --epochs 100 --upstream-dir ../Geo-Hgan-author --save-model
```

The illustrated seed is for fold 1; use the prepared patch seed for other folds. The adapter verifies six upstream source files and, when present, Git HEAD. It uses 12 channels and 32×32 patches, with **100 epochs in each of two stages**. `benchmarks/geohgan.py` documents paper/code choices and preserves the author network definitions. A checkpoint moved to another machine can be loaded with `GeoHGAN.load(path, upstream_dir=...)`.

## Unified regional maps and mineral-association evaluation

### Fit a separate full-region deployment model

```bash
python -m benchmarks.prepare_region --input data/region_full.npz --mode global --output outputs/global
python -m geopr_vae train --input outputs/global/core_train.npz --output outputs/global/core --method all --seed 20260929 --head-seed 20260929
python -m benchmarks.patch_runner --input outputs/global/patch_input.npz --output outputs/global/VAE_patch --method VAE_patch --seed 2027101004 --epochs 100
python -m benchmarks.patch_runner --input outputs/global/patch_input.npz --output outputs/global/GANomaly --method GANomaly --seed 2027101004 --epochs 100
python -m benchmarks.patch_runner --input outputs/global/patch_input.npz --output outputs/global/Skip_GANomaly --method Skip_GANomaly --seed 2027101004 --epochs 100
```

The global run uses all valid points as one query field and the deployment training/calibration split. It is separate from spatial validation: do not stitch fold predictions into the final map. Additional patch methods use the same prepared global input and seed.

### Create common grid fields

Provide the same prefiltered 500 m effective grid for every method. Its CSV contains unique integer `grid_id` and projected metre `x,y` coordinates. IDs provide a deterministic tie order. Geometry preparation validates the supplied mask and square lattice; it does not derive a study boundary or expand evaluation support.

```bash
python -m scripts.prepare_regional geometry --csv data/effective_grid.csv --crs EPSG:32649 --cell-size 500 --output outputs/grid.npz
python -m scripts.map_patch_responses --grid outputs/grid.npz --prediction VAE_patch=outputs/global/VAE_patch/predictions.npz --prediction GANomaly=outputs/global/GANomaly/predictions.npz --prediction Skip_GANomaly=outputs/global/Skip_GANomaly/predictions.npz --output outputs/patch_grid
```

Patch regional scoring uses the **matched input/reconstruction residual at pixel [16,16]**, then shared calibration, 10 km spatial aggregation, affine transformation and 12-neighbor IDW. Controlled recovery instead uses the average of the central 2×2 reconstruction pixels. These two readouts serve different evaluations and are not interchangeable.

The following complete bridge combines core and patch grids without changing their order or reranking a fixed association:

```python
from pathlib import Path
import numpy as np
from geopr_vae.mapping import grid_scores

def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

grid = arrays("outputs/grid.npz")
fields = arrays("outputs/patch_grid/association_fields.npz")
np.testing.assert_array_equal(fields["grid_id"], grid["grid_id"])
for method in ("AE", "VAE", "GeoPR-VAE"):
    point = arrays(Path("outputs/global/core") / method / "prediction.npz")
    mapped = grid_scores(point["xy"], point["affine_scores"], grid["grid_xy"])
    fields[method] = mapped["association_percentiles"]
np.savez_compressed("outputs/association_fields.npz", **fields)
```

Alternatively, `scripts.prepare_regional associations --responses FILE --output FILE` converts calibrated element response grids into the two associations. That input has `grid_id[N]`, `elements[12]`, and one `METHOD[N,12]` array per method. It does not accept raw concentrations or native scalar network scores.

### Evaluate known mineral occurrences

Supply an occurrence CSV with `site_id,grid_id,x,y,primary_target_elements`, with one unique occurrence per occupied effective grid cell. Coordinates are in the grid's metre CRS; target elements are semicolon-separated, for example `Cu;Pb;Zn`. Include the complete effective occurrence inventory because **all supplied occurrence cells are excluded from the point-background AUC background**, including occurrences outside the currently matched commodity group. The field file has `grid_id`, `association_names` and one `METHOD[N,2]` array, with columns `Cu-Pb-Zn-Ag` and `W-Sn-Mo-Bi`.

Matching follows the recorded primary commodities: Cu/Pb/Zn/Ag for the first association, W/Sn/Mo for the second. Bi remains a score channel but is not added as a new occurrence-matching label.

```bash
python -m evaluation.regional --grid outputs/grid.npz --fields outputs/association_fields.npz --occurrences data/occurrences.csv --output outputs/association
python -m scripts.make_tables association --input outputs/association/association_metrics.csv --output outputs/tables/association
```

Outputs include point-background AUC, cumulative capture over the full area, captured occurrence counts at 5/10/20% area, enrichment and occurrence density at 10%, areas required for 50/80% capture, and per-occurrence distances to the selected top-10% square-cell union. Distance is measured to cell boundaries, not merely cell centers. AUC gives half-credit for tied scores; area selection breaks ties by ascending grid ID. These metrics measure association with known occurrences in the shared effective area.

## Descriptive statistics and geological input conversion

```bash
python -m scripts.make_tables statistics --input data/concentrations.csv --output outputs/tables/statistics
```

Statistics input contains the twelve element columns in original positive concentration units (Au/Ag ppb; others ppm).

The optional geological input converter accepts verified GeoPackages and a classification mapping. It exports geological polygons and faults without drawing them. Expected layer names and the attribute-mapping schema are documented at the top of `scripts/prepare_geology.py`. Supply repeated `--package` arguments for multiple sources; a preclassified source may use `--class-field` instead of `--mapping`.

```bash
python -m scripts.prepare_geology --package data/geology_200k.gpkg --mapping data/geology_classes.json --map-crs EPSG:4214 --output outputs/geology
```

The exported coordinates use the declared geographic CRS and polygon properties contain `class_id`. Supply the verified datum and classifications rather than relabelling coordinates or inferring geological units from display colors.

## Repository contents

| Directory | Purpose |
| --- | --- |
| `geopr_vae/` | Core models, training, preprocessing, recovery, calibrated scores and IDW |
| `benchmarks/` | Controlled analytical fields, spatial roles, patch networks and paired interventions |
| `configs/` | Controlled and regional experiment settings |
| `evaluation/` | Controlled aggregation and regional occurrence metrics |
| `scripts/` | Input conversion, response mapping, aggregation and editable tables |
| `examples/` | Generated-data execution checks |
| `docs/` | Method and validation documentation |

Keep user data, upstream third-party checkouts, trained models, generated tables/figures and manuscript files outside the tracked source tree. The repository provides calculation, input-conversion and table-export code; it does not distribute the study datasets or precomputed experimental results.
