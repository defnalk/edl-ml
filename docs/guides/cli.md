# CLI reference

The `edl` command is installed by `pip install -e .`.

## `edl simulate`

Run the Gouy-Chapman-Stern solver for a single operating point and write
the sweep as JSON.

```bash
edl simulate --concentration 0.1 --valence 1 \
             --stern-thickness 3.0 --stern-permittivity 6.0 \
             --e-min -0.4 --e-max 0.4 --n-points 81 \
             --output sim.json
```

## `edl generate`

Generate a training dataset under the default sampling bounds.

```bash
edl generate --n-samples 1000 --seed 0 \
             --output data/processed/dataset.parquet
```

## `edl train`

Train the MLP surrogate with optional MLflow tracking.

```bash
edl train --data data/processed/dataset.parquet \
          --checkpoint data/models/model.pt \
          --epochs 200 --batch-size 256 --lr 1e-3 \
          --mlflow-experiment edl-ml
```

## `edl tune`

Launch an Optuna TPE study.

```bash
edl tune --data data/processed/dataset.parquet \
         --n-trials 40 --timeout 1800 \
         --storage sqlite:///edl.db \
         --out data/models/best_config.json
```

## `edl evaluate`

Compute metrics and plot parity + residual histograms on the held-out test
split.

```bash
edl evaluate --checkpoint data/models/model.pt \
             --data data/processed/dataset.parquet \
             --figures-dir reports/figures
```

## `edl predict`

Predict a full `C_dl(E)` curve at a single operating point.

```bash
edl predict --checkpoint data/models/model.pt \
            --concentration 0.05 --valence 1 \
            --e-min -0.3 --e-max 0.3 --n-points 61
```
