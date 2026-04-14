# ML pipeline

## Dataset construction

The training dataset is built on top of a Latin hypercube sample of five
physical parameters (log concentration, valence, temperature, Stern
thickness, Stern permittivity) via
[`build_capacitance_dataset`][edl_ml.data.generate.build_capacitance_dataset].
Each sample runs a full GCS sweep across 81 electrode potentials in
`[-0.4, 0.4] V`, yielding a tidy long-format DataFrame.

- Concentration is sampled on `log10(c)` because `C_d ∝ √c`.
- Valence is drawn from a discrete set `{1, 2}` rather than a continuous
  range, matching the physics.
- Splits are made *by sweep*, not by row, so no feature vector appears in
  both the train and test splits
  ([`split_by_sample`][edl_ml.data.generate.split_by_sample]).

## Surrogate architecture

The surrogate is an MLP with

- input dim 6 (5 physical features + electrode potential),
- `(128, 128, 64)` hidden widths by default,
- SiLU activations,
- optional BatchNorm and dropout,
- Kaiming initialisation, linear scalar head.

See [`MLPConfig`][edl_ml.ml.model.MLPConfig] and
[`CapacitanceMLP`][edl_ml.ml.model.CapacitanceMLP].

## Training loop

[`train_model`][edl_ml.ml.train.train_model] runs AdamW with cosine
annealing, gradient norm clipping, and early stopping. Scalers are fitted on
the training split only, preventing data leakage. An optional MLflow run
logs every hyperparameter and per-epoch metric.

```python
from edl_ml.data import SamplingBounds, build_capacitance_dataset, split_by_sample
from edl_ml.ml import MLPConfig, TrainConfig, build_loaders, train_model

df = build_capacitance_dataset(SamplingBounds(), n_samples=1000)
train, val, test = split_by_sample(df)
loaders = build_loaders(train, val, test, batch_size=256)
report = train_model(
    loaders,
    MLPConfig(input_dim=6, hidden_dims=(128, 128, 64), dropout=0.05),
    TrainConfig(max_epochs=200, learning_rate=1e-3,
                mlflow_experiment="edl-ml"),
    checkpoint_path="data/models/model.pt",
)
print(report.test_metrics)   # {'mse': ..., 'rmse': ..., 'mae': ..., 'r2': ..., 'mape': ...}
```

## Hyperparameter optimisation

[`run_optuna_study`][edl_ml.ml.tune.run_optuna_study] wraps an Optuna TPE
study over width, depth, activation, BatchNorm, dropout, learning rate, and
weight decay. Pass a persistent storage URL (`sqlite:///edl.db`) to resume.

## Interpretability

Two complementary methods ship with the package:

- [`shap_explain`][edl_ml.ml.explain.shap_explain] — KernelSHAP on the raw
  feature space, paired with
  [`plot_shap_summary`][edl_ml.viz.diagnostics.plot_shap_summary] for a
  beeswarm plot.
- [`permutation_feature_importance`][edl_ml.ml.explain.permutation_feature_importance]
  — dependency-free alternative that measures MAE increase under column
  permutation.
