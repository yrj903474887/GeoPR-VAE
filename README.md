# GeoPR-VAE

Code accompanying **GeoPR-VAE: Spatial-Contrast Residual Deep Learning for Multielement Weak Geochemical Anomaly Identification**.

Paper authors: Runjie Yang, Fan Xiao and Qiuming Cheng. Corresponding author: [Fan Xiao](mailto:xiaofan3@mail.sysu.edu.cn).

GeoPR-VAE identifies weak multielement geochemical anomalies through a target-masked conditional reference and spatial-contrast recovery. This repository includes the core method, comparison models, controlled experiments, buffered spatial validation, regional evaluation and table exports.

## Installation

Use Python 3.10 or newer in a virtual environment. Commands below assume the environment is activated and the current directory is the repository root.

```bash
python -m pip install -r requirements.txt
```

The implementation runs on **CPU** and uses one PyTorch thread per core process. The full controlled runner supports multiple worker processes. A GPU is not required by the current workflow.

For the versions actually validated, install `requirements-tested.txt` instead and read [the tested environment and computational requirements](docs/ENVIRONMENT.md). The ranges in `requirements.txt` are installation constraints; they do not imply that every version combination has been tested.

## Quick start

This example generates its own small synthetic field, fits AE, VAE and GeoPR-VAE, and checks saved-model reload, prediction consistency and paired interventions. No study data are needed.

```bash
python -m examples.smoke --save-input ../validation-output/core/synthetic.npz --report ../validation-output/core/smoke_report.json
```

It uses 8 reference epochs and 12 recovery updates for software checks. Its outputs are not paper-performance estimates. Use a new output location when repeating a run. Additional checks are described in [VALIDATION.md](docs/VALIDATION.md).

## Train, predict and create grid fields

Once the input arrays have been prepared according to the [user guide](docs/USER_GUIDE.md#input-schema):

```bash
python -m geopr_vae train --input data/region.npz --output outputs/region --method GeoPR-VAE
python -m geopr_vae predict --model outputs/region/GeoPR-VAE/model.pt --input data/query.npz --output outputs/query_prediction.npz
python -m geopr_vae grid --prediction outputs/query_prediction.npz --grid data/valid_grid.npz --output outputs/query_grid.npz
```

The core defaults retain the manuscript settings: 600 reference epochs, 300 GeoPR-VAE recovery updates and 16 posterior draws per target. The patch comparison models use their separately configured 100-epoch schedules. Full controlled and regional settings are in [configs/](configs/).

For independent spatial validation, use the explicit roles and separate seeds from the regional preparation workflow. The core training command alone uses a deployment split when no indices are supplied. Supply a complete query field to prediction; geographic partitioning changes the spatial neighborhoods and therefore the method.

## Reproduce the paper

| Documentation | Contents |
| --- | --- |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Quick checks, full synthetic experiments, regional data requirements and execution order |
| [Workflows](docs/WORKFLOWS.md) | Executable commands and recipes for controlled experiments, spatial validation, regional grid fields and tables |
| [User guide](docs/USER_GUIDE.md) | Input schemas, units, options, model outputs and expected behaviour |
| [Environment](docs/ENVIRONMENT.md) | Tested dependencies, installation and measured short-check resource use |
| [Validation](docs/VALIDATION.md) | Scope and interpretation of the supplied software checks |
| [Method configuration](docs/METHODS.md) | Model, preprocessing, training, calibration and scoring settings |

## Data and local outputs

The stream-sediment geochemical data will be made available on request to the corresponding author. Geological maps and mineral-occurrence data are available from [GeoCloud, China Geological Survey](https://geocloud.cgs.gov.cn/); ASTER elevation data are available from [Geospatial Data Cloud](https://www.gscloud.cn/sources/index?pid=302). [Reproducibility](docs/REPRODUCIBILITY.md#3-obtain-and-prepare-regional-inputs) explains the prepared inputs needed for regional replication.

Only code, configurations and documentation are distributed here. Data, trained weights and generated results remain local. The synthetic generators provide a runnable example without distributing study data. Generated arrays, checkpoints, reports, figures and manuscript files must not be committed to this repository.

## License

The code and accompanying documentation are available under the [MIT License](LICENSE). External datasets and third-party source code retain their respective terms; they are not relicensed by this repository. The optional Geo-Hgan adapter uses a separately obtained upstream checkout as described in [WORKFLOWS.md](docs/WORKFLOWS.md#additional-comparison-networks).
