# Running validation checks

Run checks from the repository root in the same environment used for training.

```bash
python -m examples.smoke --save-input outputs/synthetic.npz --report outputs/smoke_report.json
python -m benchmarks.test_smoke
python -m scripts.smoke_evaluation --help
```

The core smoke example exercises AE, VAE and GeoPR-VAE on generated inputs, checkpoint reload, batched reference prediction, target masking, training-fitted preprocessing, paired interventions and grid interpolation. It uses a short training schedule for software checks.

The benchmark checks exercise the comparison networks, analytical fields, spatial splits and intervention metrics. The evaluation checks exercise occurrence capture, area efficiency, nearest-area distances, aggregation and figure generation using generated examples.

Generated arrays, checkpoints, figures and reports belong in a local output directory and are excluded from version control. Small execution checks do not replace the configured full experiments. Library and hardware differences may affect floating-point results.
