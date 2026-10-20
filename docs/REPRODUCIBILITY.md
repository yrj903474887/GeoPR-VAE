# Reproducing GeoPR-VAE

Run commands from the repository root. Use the [tested environment](ENVIRONMENT.md)
and preserve the model settings in `configs/paper_experiments.json` and
`configs/regional_experiments.json`.

## Choose a workflow

| Workflow | Required inputs | What it establishes |
| --- | --- | --- |
| Quick software checks | Generated locally by the supplied Python examples | Installation, finite outputs, model reload, data-role checks and evaluation calculations |
| Complete controlled experiments | Analytical fields generated locally by `benchmarks.controlled` | The six-method confirmation experiment under the paper's full schedules |
| Regional validation and mapping | Study inputs obtained and prepared as described below | Five-fold spatial validation, regional predictions, occurrence evaluation and maps |

The quick checks deliberately use small inputs and abbreviated training.
Their numerical outputs are not the results reported in the paper. Full
experimental settings remain unchanged; a short check cannot be aggregated
into a complete confirmation experiment.

## 1. Quick software checks

```bash
python -m examples.smoke --save-input ../validation-output/core/synthetic.npz --report ../validation-output/core/smoke_report.json
python -m benchmarks.test_smoke
python -m scripts.smoke_evaluation --output ../validation-output/evaluation_smoke --plots --font-family "DejaVu Sans"
python -m benchmarks.controlled --config configs/paper_experiments.json --phase confirmation --output ../validation-output/controlled_smoke --workers 1 --limit 1 --smoke
```

Use fresh output directories for repeat runs. The first command generates a
160-point, twelve-element field and checks the core models. The evaluation
example creates local toy inputs, reports and plots. The last command exercises
one controlled task with the runner's existing smoke option. It does not alter
the experiment configuration file.

[Validation](VALIDATION.md) describes the checks and their interpretation.
[Environment](ENVIRONMENT.md) records the environment actually tested and the
measured resource use. Generated data, reports, weights and plots remain local.

## 2. Complete controlled experiments

```bash
python -m benchmarks.controlled --config configs/paper_experiments.json --phase confirmation --output outputs/controlled --workers 8
python -m benchmarks.aggregate --config configs/paper_experiments.json --phase confirmation --run outputs/controlled --output outputs/controlled_tables
```

The confirmation matrix has 720 fitted tasks per method and 4,320 tasks across
the six methods. Each task evaluates 144 paired interventions. Core reference
networks use 600 epochs, GeoPR-VAE recovery uses 300 updates, and the three patch
comparators use 100 epochs. The ten confirmation seeds and the development
seeds are separate. Keep their results in separate directories.

The aggregator validates completeness and rejects smoke outputs, missing tasks
and duplicated tasks. [Workflows](WORKFLOWS.md#complete-controlled-experiment-matrix)
gives the full procedure. [Figures](FIGURES.md) maps the resulting tables to
the current manuscript figures and documents the final presentation steps.

## 3. Obtain and prepare regional inputs

The stream-sediment geochemical data will be made available on request.
Contact the corresponding author, Fan Xiao, at
[xiaofan3@mail.sysu.edu.cn](mailto:xiaofan3@mail.sysu.edu.cn).

The geological maps and mineral-occurrence data are available from the
[GeoCloud platform of the China Geological Survey](https://geocloud.cgs.gov.cn/).
ASTER elevation data are available from the
[Geospatial Data Cloud](https://www.gscloud.cn/sources/index?pid=302).
These links identify the data providers; they are not automatic download
endpoints for the paper's prepared inputs. Obtain the relevant products under
their providers' terms and follow the preparation instructions.

| Prepared input | Required information |
| --- | --- |
| Regional point `.npz` | `log10[N,12]`, `xy[N,2]`, `terrain[N,3]`, `groups[N]`, `ids[N]`; optional terrain-quality `valid[N]` |
| Effective-grid CSV | Unique `grid_id`, projected metre `x,y`, the common 500 m grid and the paper's sampling-support mask |
| Evaluation occurrence CSV | `site_id,grid_id,x,y,primary_target_elements`, with the complete effective inventory and semicolon-separated commodities |
| Cartographic inputs | Converted geology and faults, class mapping, geological occurrence coordinates, display boundary and georeferenced hillshade |

The point input uses element order `Au, Ag, As, Bi, Cu, F, Mo, Pb, Sb, Sn, W, Zn`.
Au and Ag are in ppb; the remaining elements are in ppm before taking `log10`.
The regional projected coordinates use EPSG:32649 in metres. All sample arrays
must retain identical row order and stable sample IDs. See the
[user guide](USER_GUIDE.md#input-schema) for complete shapes, units and options.

To reproduce the paper's spatial folds, retain the complete coordinate/ID frame
until `benchmarks.prepare_region` has constructed the folds. Apply the terrain
validity mask in the documented order. The valid regional sample count is 4,651;
the effective evaluation grid contains 75,121 cells. These counts are useful
cross-checks, but matching counts alone does not establish identical inputs.

Retain the source-product identifiers, acquisition dates, processing choices,
file hashes and coordinate transformations with your local input files.
Prepared study data and fixed evaluation masks are not bundled here. Exact
regional replication depends on obtaining the corresponding study inputs;
the synthetic workflows remain usable without them.

## 4. Execute regional workflows

Follow these sections of [WORKFLOWS.md](WORKFLOWS.md) in order:

1. Prepare all five spatial folds with both 4 km and 8 km exclusion buffers.
2. Train using each preparation output's reference, recovery and patch seeds.
3. Predict the original and 144 paired weak-intervention fields and aggregate
   the fold-level evaluation.
4. Fit the separate full-region deployment models.
5. Interpolate onto the common effective grid and evaluate the complete
   matching occurrence inventory.
6. Generate manuscript panels using the [figure mapping](FIGURES.md).

The deployment split and buffered validation serve different purposes.
Retain their distinct seeds and roles. Regional maps use the full-region model;
they are not assembled by joining the held-out fold predictions.

The existing commands and Python recipes are the executable interfaces.
There is no separate one-command regional reproduction driver.

## Local outputs and source publication

This repository distributes source code, configurations and documentation.
Examples generate their inputs locally. Study data, third-party checkouts,
trained weights, arrays, tables, reports, figures and manuscript files are
excluded from the tracked source tree.

Keep all methods on the same supplied inputs and preserve the configured
training schedules, seeds, preprocessing, scoring and evaluation rules.
Computational results can differ slightly across library versions and
hardware; use the tested versions when investigating such differences.
