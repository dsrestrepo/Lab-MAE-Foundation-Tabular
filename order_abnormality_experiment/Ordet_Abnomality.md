# Lab Ordering and Abnormality Experiment

This experiment asks whether the Lab-MAE embedding of the current lab state contains
information about what will happen on the next lab day.

The dataset already has both current and next-day lab values in the same row:

- Current value: `npval_<lab_id>`
- Current time: `nptime_<lab_id>`
- Next-day value: `npval_last_<lab_id>`
- Next-day time: `nptime_last_<lab_id>`

For example:

- `npval_50971` is the current Potassium value.
- `npval_last_50971` is the next-day Potassium value, if it exists.

## Target Labs

We only evaluate labs where we have simple normal/abnormal ranges:

- Bicarbonate
- Creatinine
- Potassium
- Sodium
- Urea Nitrogen
- Hemoglobin
- Platelet Count
- White Blood Cells

For each target lab and each row, we create:

- `ordered`: `1` if `npval_last_<lab_id>` is present, otherwise `0`
- `abnormal`: `1` if the next-day value is outside the normal range, `0` if normal, and missing if not ordered

## Important Leakage Detail

The Lab-MAE checkpoint expects the full 400-column input shape, including the `last_`
columns. However, those `last_` columns contain the answers for this experiment.

So before extracting embeddings, `build_order_abnormality_dataset.py` keeps the full
shape but sets every `*_last_*` column to `NaN`. This prevents the embedding from
seeing the next-day values.

## Files

### `build_order_abnormality_dataset.py`

Creates the experiment dataset.

It:

1. Loads `../data/X_test.csv`.
2. Builds labels from `npval_last_<lab_id>`.
3. Masks all `last_` columns before embedding extraction.
4. Extracts Lab-MAE embeddings.
5. Pools column-level Lab-MAE embeddings into one row-level vector `z`.
6. Saves a long-format dataset with one row per patient/sample and target lab.

Main outputs:

- `data/order_abnormality_dataset.csv`
- `data/order_abnormality_summary.csv`

### `run_order_abnormality_experiment.py`

Runs three prediction analyses.

## Analysis 1: Next Order Prediction

Question:

> Can the current Lab-MAE embedding predict whether a lab will be measured next day?

Target:

```text
ordered = 1 if npval_last_<lab_id> exists
```

Model:

```text
P(ordered | z)
```

Outputs:

- `order_auroc`
- `order_brier`
- `bar_order_auroc.png`
- `bar_order_brier.png`
- `roc_order_prediction.png`
- `calibration_order_prediction.png`

Interpretation:

- AUROC above 0.5 means the embedding can rank rows by likelihood of next-day ordering.
- Lower Brier score means better probability accuracy.
- Calibration close to the diagonal means predicted order probabilities match observed order rates.

## Analysis 2: Next Abnormality Among Ordered Labs

Question:

> Among labs that were actually measured next day, can the current Lab-MAE embedding predict whether the result will be abnormal?

Target:

```text
abnormal = 1 if npval_last_<lab_id> is outside the normal range
```

Rows used:

```text
only rows where ordered == 1
```

Model:

```text
P(abnormal | ordered = 1, z)
```

Outputs:

- `abnormal_auroc`
- `abnormal_auprc`
- `bar_abnormal_auroc.png`
- `bar_abnormal_auprc.png`
- `roc_abnormal_if_ordered.png`
- `calibration_abnormal_if_ordered.png`

Interpretation:

- This measures abnormality prediction conditional on the lab being observed.
- AUPRC is useful when abnormal cases are relatively rare.
- Calibration shows whether predicted abnormal probabilities match observed abnormal rates among ordered labs.

## Analysis 3: Joint Observed-And-Abnormal Prediction

Question:

> Across all rows, can the embedding predict whether the lab will both be measured and abnormal next day?

Target:

```text
observed_and_abnormal = 1 if ordered == 1 and abnormal == 1
```

Predicted probability:

```text
P(ordered and abnormal | z)
  = P(ordered | z) * P(abnormal | ordered = 1, z)
```

In the output this is:

```text
expected_abnormal = pred_order * pred_abnormal
```

Outputs:

- `joint_auroc`
- `joint_auprc`
- `joint_brier`
- `bar_joint_auroc.png`
- `bar_joint_auprc.png`
- `bar_joint_brier.png`
- `roc_joint_observed_abnormal.png`
- `calibration_joint_observed_abnormal.png`

Interpretation:

- This is the cleanest single outcome for "will we observe an abnormal result next day?"
- AUROC/AUPRC evaluate ranking of the joint event.
- Brier/calibration evaluate whether `expected_abnormal` behaves like a real probability.

## Yield Analysis

This is an additional analysis using the order prediction.

It:

- Computes `expected_abnormal = pred_order * pred_abnormal`.
- Groups by predicted ordering deciles.
- Checks whether higher predicted ordering probability corresponds to higher observed abnormal yield.

Main outputs are written to `results/`:

- `metrics.csv`
- `predictions.csv`
- `order_decile_abnormal_yield.csv`
- `order_decile_abnormal_yield.png`
- `calibration_curves.csv`
- `roc_curves.csv`
- bar plots for AUROC/AUPRC/Brier metrics
- ROC and calibration plots for the three analyses

## How To Run

Run all commands from this directory:

```bash
cd "order_abnormality_experiment"
```

First, check whether the labels have enough cases. This does not extract embeddings:

```bash
python3 build_order_abnormality_dataset.py \
  --max-rows 5000 \
  --skip-embeddings
```

Then build the full dataset with Lab-MAE embeddings:

```bash
python3 build_order_abnormality_dataset.py \
  --max-rows 5000
```

If using Apple Silicon/Mac GPU, you can force MPS:

```bash
python3 build_order_abnormality_dataset.py \
  --max-rows 5000 \
  --device mps
```

Then run the experiment:

```bash
python3 run_order_abnormality_experiment.py \
  --input data/order_abnormality_dataset.csv \
  --output-dir results
```

Since those are the defaults, this shorter command is equivalent:

```bash
python3 run_order_abnormality_experiment.py
```

## Notes

- The dataset created with `--skip-embeddings` cannot be used by the experiment script,
  because it does not contain `z_0`, `z_1`, etc.
- `--max-rows` is useful for quick testing. Remove it or increase it for the full run.
