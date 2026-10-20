# Tested environment

The checks below passed on 20 October 2026 in a newly created Python virtual environment. It had no access to system site packages or the user site. The environment was installed from `requirements-tested.txt`, using versions within the installation constraints in `requirements.txt`; no research algorithm, seed, training schedule or experiment configuration was changed.

`requirements.txt` describes installation constraints. [`requirements-tested.txt`](../requirements-tested.txt) records the eight direct dependencies and their complete transitive dependency closure: 28 exact package versions for this CPU validation environment. It is a Windows/Python 3.12 snapshot, not a lockfile tested on every operating system or Python version.

## Hardware and software

| Item | Tested value |
| --- | --- |
| Operating system | Windows 11, build 22631, x86-64 |
| Python | CPython 3.12.14, 64-bit |
| Installer | pip 25.0.1 |
| CPU | AMD Ryzen 7 7700; 8 physical cores, 16 logical processors |
| Installed system memory | 33,399,156,736 bytes, approximately 31.1 GiB |
| PyTorch build | CPU-only; no CUDA runtime or GPU used |

| Direct dependency | Tested version |
| --- | --- |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| PyTorch | 2.14.1+cpu |
| pandas | 2.3.3 |
| pyproj | 3.8.0 |
| Shapely | 2.2.0 |
| Rasterio | 1.5.2 |
| Fiona | 1.10.1 |

All eight direct dependencies imported successfully, and `python -m pip check` reported no broken requirements. The installed packages match the 28 recorded versions. The installed wheels expose GDAL 3.12.2 through Rasterio, GDAL 3.9.1 through Fiona, and PROJ 9.8.1 through pyproj. No separate system GDAL installation was used.

## Recreate the tested CPU environment

Use Python 3.12 and run from the repository root:

```bash
python -m venv ../geopr-venv
```

Activate it on Windows PowerShell:

```powershell
..\geopr-venv\Scripts\Activate.ps1
```

Then install the recorded versions:

```bash
python -m pip install -r requirements-tested.txt
python -m pip check
```

The tested requirements file includes the official PyTorch CPU wheel index and pins the `+cpu` build. For a different platform, start with the ranges in `requirements.txt` and select an appropriate PyTorch build. Linux, macOS, CUDA execution and other Python versions were not tested in this validation.

## Checks performed

All commands were run from the repository root. Output directories and temporary files were outside the source repository. The existing smoke schedules were used unchanged.

```bash
python -m examples.smoke --save-input ../validation-output/core/synthetic.npz --report ../validation-output/core/smoke_report.json
python -m benchmarks.test_smoke
python -m scripts.smoke_evaluation --output ../validation-output/evaluation
python -m benchmarks.controlled --output ../validation-output/controlled --limit 1 --smoke
```

| Check | Outcome | Wall time (s) | Sampled peak process-tree RSS (MiB) |
| --- | --- | ---: | ---: |
| `examples.smoke` | PASS: AE, VAE and GeoPR-VAE structural checks | 2.63 | 325.6 |
| `benchmarks.test_smoke` | PASS: five tests, including comparison-model checks | 3.10 | 424.2 |
| `scripts.smoke_evaluation` | PASS: capture, area-efficiency and distance calculations | 0.40 | 70.1 |
| `benchmarks.controlled --limit 1 --smoke` | PASS: one configured AE task, one epoch and two probes | 3.65 | 510.0 |

The core example uses its existing eight reference epochs and twelve recovery updates. The controlled command retains the runner's default worker setting of eight; `--limit 1` restricts it to the first task. These reduced schedules are explicitly selected by the existing smoke entry points. They do not alter the full experiment configuration.

Wall time includes interpreter startup. Memory is the maximum observed sum of resident memory for the command's process and its descendants, sampled every 50 ms using an external psutil 7.2.2 monitor. It is a sampled working-set measurement, not a guaranteed instantaneous peak, Python-only allocation count or system-wide memory measurement. Shared pages can be counted in more than one process. The monitor is not a package dependency. These small-check measurements do not estimate the resources needed for full regional training or the complete experiment matrix.

The checks use generated inputs and validate execution, structural invariants and evaluation calculations. They do not reproduce the paper's reported performance or substitute for its complete experiments. Raw logs, generated inputs, checkpoints and measurement reports remain outside the repository.
