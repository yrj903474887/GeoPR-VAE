# Manuscript figures

The default drawing commands now use the current manuscript numbers. Each
experimental panel is one PDF with its own legend. Panel letters and captions
are added in LaTeX, not drawn into the PDF. The scripts consume the outputs and
prepared geographic inputs described in [WORKFLOWS.md](WORKFLOWS.md). They do
not change the algorithms, spatial splits, training schedules or evaluation
parameters.

No study data, trained weights, precomputed statistics or finished figures are
distributed in this repository. Put inputs and generated figures outside the
tracked source tree; the examples below use sibling directories `../paper-input`
and `../paper-output`.

## Figure-to-code mapping

| Current file(s) | Content and panel order | Drawing entry point | Earlier intermediate name(s) |
| --- | --- | --- | --- |
| `Fig01.pdf` | Method architecture | Author-drawn vector schematic; not a computed experiment and no automatic drawing script is supplied | `fig01.pdf` |
| `Fig02.pdf` | Geological map, locator and integrated geological/occurrence legends | `plotting.geology`; optional component assembly with `scripts.figures_finalize geology` | `fig02_b.pdf` + `fig02_a.pdf` + `fig02_occurrence_legend.pdf` |
| `Fig03a.pdf` | Fixed five spatial folds | `plotting.spatial_folds` | `fig03_a.pdf` |
| `Fig03b.pdf`, `Fig03c.pdf` | Held-out fold roles at 4 km and 8 km, respectively | `plotting.spatial_folds` | `fig03_b.pdf`, `fig03_c.pdf` + shared role legend |
| `Fig04a.pdf`, `Fig04b.pdf` | Recovery RMSE; mean casewise amplitude-gain error | `plotting.paper_figures recovery` | `fig04_a.pdf`, **`fig04_c.pdf`** |
| `Fig05a.pdf`, `Fig05b.pdf` | Recovery RMSE by noise condition, at 4 km and 8 km | `plotting.paper_figures noise` | `fig05_noise_4km.pdf`, `fig05_noise_8km.pdf` |
| `Fig05c.pdf`, `Fig05d.pdf` | Amplitude-gain error by enrichment amplitude, at 4 km and 8 km | `plotting.paper_figures amplitude` | `fig05_gain_4km.pdf`, `fig05_gain_8km.pdf` |
| `Fig06a.pdf`, `Fig06b.pdf` | Weak-injection AUROC; core recall in the top 10% | `plotting.paper_figures weak` | `fig_weak_delta_auc.pdf`, `fig_weak_delta_recall.pdf` |
| `Fig07a.pdf`–`Fig07f.pdf` | W–Sn–Mo–Bi regional response maps | `plotting.regional_maps` | `fig6_WSnMoBi_*` + shared legend |
| `Fig08a.pdf`–`Fig08f.pdf` | Cu–Pb–Zn–Ag regional response maps | `plotting.regional_maps` | `fig6_CuPbZnAg_*` + shared legend |
| `Fig09a.pdf`, `Fig09b.pdf` | Cumulative occurrence capture: Cu–Pb–Zn–Ag, W–Sn–Mo–Bi | `plotting.paper_figures capture` | `fig7_Cu10.pdf`, `fig7_W12.pdf` |
| `Fig10a.pdf`, `Fig10b.pdf` | Distance to the selected anomaly area: Cu–Pb–Zn–Ag, W–Sn–Mo–Bi | `plotting.paper_figures distance` | `fig8_Cu10.pdf`, `fig8_W12.pdf` |

The six map panels are always ordered **AE, VAE, Patch VAE, GANomaly,
Skip-GANomaly, GeoPR-VAE**. The old `fig04_c` is now `Fig04b`; it must not be
confused with a different diagnostic. `Cu10` and `W12` are internal association
keys in the evaluation output, not current figure numbers.

## Fonts and output layout

Use `Times New Roman`, the default, for the manuscript figures. A licensed local
installation is required; font files are not included. Supply `--font-dir` when
the font is not installed system-wide. `--font-family "DejaVu Sans"` is available
for portable software demonstrations, but changes typography and is not the
manuscript appearance.

The publication layouts preserve:

- Pure black axes frames and statistical bar/box boundaries.
- A separate, two-row, three-column legend in each Fig04–06 panel.
- Two-row, two-column role keys below Fig03b/c, aligned with the map frame.
- A 60-point integrated footer in each Fig07/08 panel. The existing map is
  translated without scaling and the original color ramp is reused.
- A 12:11 page ratio for Fig09 panels and a two-row, three-column legend aligned
  with the axes.

PDF output is the default. Quantitative plots use Matplotlib for optional PNGs.
Geology, spatial folds and regional maps render their completed PDFs for `--png`;
these optional previews require Poppler's `pdftoppm` on `PATH`. A missing Poppler
installation does not prevent PDF generation when `--png` is omitted.

## Quantitative panels: Fig04–06 and Fig09–10

After running the aggregation and regional evaluation steps in WORKFLOWS:

```bash
python -m plotting.paper_figures recovery --input ../paper-output/controlled_tables/base_means.csv --output ../paper-output/figs
python -m plotting.paper_figures noise --input ../paper-output/controlled_tables/noise_base_means.csv --output ../paper-output/figs
python -m plotting.paper_figures amplitude --input ../paper-output/controlled_tables/amplitude_base_means.csv --output ../paper-output/figs
python -m plotting.paper_figures weak --input ../paper-output/weak_tables/fold_metrics.csv --output ../paper-output/figs
python -m plotting.paper_figures capture --input ../paper-output/association/association_capture_curves.npz --metrics ../paper-output/association/association_metrics.csv --output ../paper-output/figs
python -m plotting.paper_figures distance --input ../paper-output/association/association_per_site.csv --metrics ../paper-output/association/association_metrics.csv --output ../paper-output/figs
```

Fig04/05 require ten distinct realization means for every method, buffer and
displayed stratum. Fig05c/d retain the existing bootstrap confidence intervals;
they are not plain bars. Fig06 requires the five fold means for each method and
buffer. The plotting code checks capture curves against the summary `N10` and
checks distance means against the per-occurrence table. Do not substitute
individual injections for realization means or training results for held-out
results.

## Spatial folds: Fig03

Supply the frozen assignments as a CSV:

| Column | Meaning |
| --- | --- |
| `sample_id` | Unique sample identifier |
| `x`, `y` | Original projected coordinates, metres in UTM zone 49N |
| `fold` | Fixed fold number, 1–5 |
| `role_4km`, `role_8km` | `T` training, `C` calibration, `E` evaluation or `X` excluded |

Only the manuscript's valid regional subset belongs in this table. The two role
columns refer to the same held-out fold. The plotting command validates that
evaluation-role records match that fold and derives every displayed count from
the table; it never regenerates a split.

```bash
python -m plotting.spatial_folds --input ../paper-input/spatial_roles.csv --heldout-fold 1 --output ../paper-output/figs
```

This is the Maoming–Yangjiang publication layout with fixed coordinate ticks,
page size, point size and labels. Coordinates outside the expected region are
rejected explicitly. For a different study area, adapt the cartographic layout;
do not change the assignments to make them fit. The drawing keeps equal map
scales and pads the shorter display extent so that the legend and frame stay
aligned. All sample markers remain vectors.

An optional path updates **original-size, unmerged** Fig03b/c PDFs while leaving
their map contents intact:

```bash
python -m scripts.figures_finalize folds --panel-b ../paper-input/components/fig03_b.pdf --panel-c ../paper-input/components/fig03_c.pdf --roles ../paper-input/spatial_roles.csv --output ../paper-output/figs
```

Use exactly the assignments used to draw those components. This path removes
their old abbreviated count footer; it rejects incompatible page sizes rather
than rescaling or guessing. It does not create Fig03a.

## Regional maps: Fig07 and Fig08

```bash
python -m plotting.regional_maps --grid ../paper-output/grid.npz --fields ../paper-output/association_fields.npz --occurrences ../paper-input/occurrences.csv --boundary ../paper-input/study_boundary_wgs84.geojson --output ../paper-output/figs
```

The input grid, association-field order and occurrence matching are the same as
in regional evaluation. Each map and its integrated legend are saved together;
there is no shared legend to add in LaTeX. Display-only mineral symbol grouping
does not alter the scoring fields or matched occurrence inventory. For older
documents only, `--legacy-names` retains the former `fig6_*` maps and separate
shared legends.

## Geology: Fig02

Prepare the geological polygons/faults using `scripts.prepare_geology` as
documented in WORKFLOWS. The plotting command uses their supplied native
geographic CRS, the separate geological occurrence table, WGS84 display
boundaries and a georeferenced hillshade. Supply the verified class mapping and
datum transformation used for the intended map; software cannot infer these
from a PDF or from an unlabelled coordinate array.

```bash
python -m plotting.geology --geology ../paper-output/geology/geology.geojson --faults ../paper-output/geology/faults.geojson --legend ../paper-output/geology/legend.json --occurrences ../paper-input/geological_occurrences.csv --boundary ../paper-input/study_boundary_wgs84.geojson --province ../paper-input/guangdong_wgs84.geojson --hillshade ../paper-input/hillshade.tif --map-crs EPSG:4214 --output ../paper-output/figs
```

This command directly creates `Fig02.pdf`, including both legends and the
locator. The main page is 547.2 × 475.2 points. Add `--components` to retain the
map without the locator, the standalone locator and the separate occurrence
legend as well. Those extra component files are working files, not additional
manuscript figures.

The original-size component assembly is also available for an archived
publication map:

```bash
python -m scripts.figures_finalize geology --map ../paper-input/components/fig02_b.pdf --locator ../paper-input/components/fig02_a.pdf --occurrence-legend ../paper-input/components/fig02_occurrence_legend.pdf --output ../paper-output/figs
```

This last command uses the manuscript's fixed locator placement and requires
its original page geometry. It must not be applied to a main map that already
contains that locator or occurrence legend. For changed geographic extents,
use the direct `plotting.geology` command, which computes the inset relative to
the current map axes.

## What the checks establish

The public workflow provides executable drawing and vector-assembly code,
not proof that an arbitrary supplied dataset reproduces the paper. The
publication input tables, geological classifications, split assignments and
the licensed Times font are needed to regenerate the study figures.

Local checks of the publication component assembly reproduced Fig02 and Fig03b/c
pixel-for-pixel at 288 dpi and all twelve Fig07/08 panels at 150 dpi.
The direct Fig03 renderer was also checked with the original assignments and
retained its points, counts, vectors and layout. Synthetic CLI checks exercise
the regional-map and geology drawing paths; they are software checks, not a
repeat of the scientific experiments. See [ENVIRONMENT.md](ENVIRONMENT.md) for
the separately recorded software environment and execution checks.
