# Running validation checks

Run checks from the repository root in the same environment used for training.

```bash
python -m examples.smoke --save-input ../validation-output/core/synthetic.npz --report ../validation-output/core/smoke_report.json
python -m benchmarks.test_smoke
python -m scripts.smoke_evaluation --output ../validation-output/evaluation_smoke --plots --font-family "DejaVu Sans"
python -m benchmarks.controlled --config configs/paper_experiments.json --phase confirmation --output ../validation-output/controlled_smoke --workers 1 --limit 1 --smoke
```

The core smoke example exercises AE, VAE and GeoPR-VAE on generated inputs, checkpoint reload, batched reference prediction, target masking, training-fitted preprocessing, paired interventions and grid interpolation. It uses a short training schedule for software checks.

The benchmark checks exercise the comparison networks, analytical fields, spatial splits and intervention metrics. The evaluation checks exercise occurrence capture, area efficiency, nearest-area distances, aggregation and figure generation using generated examples.

The controlled smoke command runs one task with the runner's existing short
schedule and two probes. It checks the execution path only. Full confirmation
aggregation deliberately rejects its output.

Read [ENVIRONMENT.md](ENVIRONMENT.md) for the actual tested environment and
measured outcomes. A check's presence in this file does not establish that the
full paper experiment has been rerun. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
and [WORKFLOWS.md](WORKFLOWS.md) for full experiments and regional inputs.

Generated arrays, checkpoints, figures and reports belong in a local output directory and are excluded from version control. Small execution checks do not replace the configured full experiments. Library and hardware differences may affect floating-point results.
