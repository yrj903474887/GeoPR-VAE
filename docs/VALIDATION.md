# Validation

Validated on 2026-10-09 with Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and PyTorch 2.14.0+cpu.

- `python -m examples.smoke`: passed for AE, VAE and GeoPR-VAE. Checks finite outputs, saved-model reload, reference prediction across different internal batch sizes, ID-keyed target masking, constant-residual unit gain with fixed references, training-only preprocessing, nonzero learned nuisance, paired complete-pipeline intervention and grid interpolation.
- All three command-line workflows completed the full default training schedule on the generated 160-row example: 600 reference epochs, plus 300 GeoPR-VAE recovery updates. This checks schedule execution, not scientific accuracy on a small field.
- The `predict` command loaded a trained checkpoint and scored a separate 24-row query field; `grid` interpolated its scores onto six supplied locations successfully.
- Numerical extraction audit against the original source files: identical preprocessing arrays, network initial weights, objective terms, posterior references, recovery parameters after 12 updates, recovery predictions, and both calibration-MAD and negative-half score outputs for identical inputs and seeds. Maximum observed difference was zero for these comparisons in the tested environment.
- The portable regional split reproduced the original 3,721 / 930 index partition from the original 4,651-sample geometry and identifiers. Those measurements and identifiers are not included in this repository.

These checks establish runnable code and agreement with the extracted components on the tested inputs. They do not rerun the paper's full controlled confirmation, buffered regional validation or mineral-association analysis. Exact floating-point equality across different library versions and hardware is not asserted. The smoke example remains available without regional data; the original-source comparison requires the private research archive.
