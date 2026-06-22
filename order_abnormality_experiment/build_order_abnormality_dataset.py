import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


# Labs where we already have a simple normal/abnormal definition.
# Keys are MIMIC lab item IDs used in columns like npval_50971.
NORMAL_RANGES = {
    50882: ("Bicarbonate", 23.0, 28.0),
    50912: ("Creatinine", 0.7, 1.3),
    50971: ("Potassium", 3.5, 5.0),
    50983: ("Sodium", 136.0, 145.0),
    51006: ("Urea Nitrogen", 8.0, 20.0),
    51222: ("Hemoglobin", 12.0, 18.0),
    51265: ("Platelet Count", 150.0, 450.0),
    51301: ("White Blood Cells", 4.0, 11.0),
}

ID_COLS = ["first_race", "chartyear", "hadm_id"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build lab ordering/abnormality dataset using current-row Lab-MAE embeddings."
    )
    parser.add_argument("--input", default=str(REPO_DIR / "data" / "X_test.csv"))
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "data" / "order_abnormality_dataset.csv"))
    parser.add_argument("--summary-output", default=str(EXPERIMENT_DIR / "data" / "order_abnormality_summary.csv"))
    parser.add_argument("--save-path", default=str(PROJECT_DIR / "100_Labs_Train_0.25Mask_L_V3"))
    parser.add_argument("--weights", default=str(PROJECT_DIR / "100_Labs_Train_0.25Mask_L_V3" / "epoch390_checkpoint"))
    parser.add_argument("--norm-parameters", default=str(PROJECT_DIR / "100_Labs_Train_0.25Mask_L_V3" / "norm_parameters.pkl"))
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--min-current-observed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or mps. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only create labels/counts. Useful before installing Lab-MAE dependencies.",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    return parser.parse_args()


def abnormal_label(value, low, high):
    if pd.isna(value):
        return np.nan
    return int(value < low or value > high)


def resolve_device(device):
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_imputer(args, dim):
    from MAEImputer import ReMaskerStep

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    imputer = ReMaskerStep(
        dim=dim,
        mask_ratio=args.mask_ratio,
        max_epochs=1,
        save_path=args.save_path,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        weigths=args.weights,
        device=device,
    )

    with open(args.norm_parameters, "rb") as file:
        imputer.norm_parameters = pickle.load(file)

    return imputer


def pooled_embeddings(imputer, X, batch_size):
    embeddings = imputer.extract_embeddings(X, eval_batch_size=batch_size).numpy()
    observed = X.notna().to_numpy(dtype=np.float32)

    pooled = []
    for i in range(embeddings.shape[0]):
        mask = observed[i].astype(bool)
        if mask.sum() == 0:
            pooled.append(np.full(embeddings.shape[-1], np.nan))
        else:
            # Lab-MAE returns one embedding per input column. We pool observed
            # current columns into one row-level representation z.
            # Important: do not multiply missing embeddings by 0, because
            # NaN * 0 is still NaN. Select observed columns before averaging.
            pooled.append(np.nanmean(embeddings[i][mask], axis=0))

    return np.asarray(pooled)


def get_target_itemids(df):
    itemids = []
    for itemid in NORMAL_RANGES:
        current_col = f"npval_{itemid}"
        future_col = f"npval_last_{itemid}"
        if current_col in df.columns and future_col in df.columns:
            itemids.append(itemid)
    return itemids


def make_model_input(df, model_cols):
    X = df[model_cols].copy()

    # The dataset already contains next-day answers in npval_last_* columns.
    # We'll re asign future columns "last" to NaN to avoid data leakage
    last_cols = [col for col in X.columns if "_last_" in col]
    X[last_cols] = np.nan
    return X


def build_labels(df, target_itemids, current_value_cols, min_current_observed):
    rows = []
    # Count how many current lab values are observed for each row.
    current_observed = df[current_value_cols].notna().sum(axis=1)

    for source_row, row in df.iterrows():
        # Verify that the current row has at least min_current_observed lab values observed.
        if current_observed.loc[source_row] < min_current_observed:
            continue
        
        patient_meta = {
            "source_row": source_row,
            "hadm_id": row.get("hadm_id", np.nan),
            "first_race": row.get("first_race", np.nan),
            "chartyear": row.get("chartyear", np.nan),
            "current_observed": int(current_observed.loc[source_row]),
        }

        # for each target lab, create a row with the current value, next-day value, and abnormality label.
        for itemid in target_itemids:
            lab_name, low, high = NORMAL_RANGES[itemid]
            current_value_col = f"npval_{itemid}"
            current_time_col = f"nptime_{itemid}"
            next_value_col = f"npval_last_{itemid}"
            next_time_col = f"nptime_last_{itemid}"

            next_value = row[next_value_col]

            # ordered means this lab exists in the precomputed next-day column.
            # If it was not ordered/measured next day, abnormality is undefined.
            ordered = int(pd.notna(next_value))
            abnormal = abnormal_label(next_value, low, high) if ordered else np.nan

            rows.append(
                {
                    **patient_meta,
                    "lab_itemid": itemid,
                    "lab_name": lab_name,
                    "current_lab_value": row.get(current_value_col, np.nan),
                    "current_lab_time": row.get(current_time_col, np.nan),
                    "ordered": ordered,
                    "abnormal": abnormal,
                    "next_lab_value": next_value if ordered else np.nan,
                    "next_lab_time": row.get(next_time_col, np.nan) if ordered else np.nan,
                    "normal_low": low,
                    "normal_high": high,
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # load the test dataset
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"Input shape after row limit: {df.shape}")

    # Get lab items and time
    model_cols = [col for col in df.columns if col not in ID_COLS]
    # Get current-value columns (npval_*) that are not next-day (npval_last_*)
    current_value_cols = [
        col for col in model_cols if col.startswith("npval_") and "_last_" not in col
    ]
    # Get columns where we have a normal range defined
    target_itemids = get_target_itemids(df)
    print(f"Target labs with ranges and current/last columns: {target_itemids}")

    # Get dataset masking next day values to avoid data leakage.
    X_current = make_model_input(df, model_cols)

    # Build the dataset with one row per patient-lab pair, including current value, next-day value, and abnormality label.
    # e.g.
    # source_row | hadm_id  | chartyear | current_observed | lab_itemid | lab_name | current_lab_value | current_lab_time | ordered | abnormal | next_lab_value | next_lab_time | normal_low | normal_high | Embedding_row
    #0          | 1234      | 2008      | 5                | 50971      | Potassium | 4.2               | 2008-01-01 08:00 | 1       | 0        | 4.5            | 2008-01-02 08:00 | 3.5        | 5.0.    | [0,2, 12, 3, -1]
    #1          | 12345     | 2008      | 5                | 50983      | Sodium    | 140.0             | 2008-01-01 08:00 | 1       | 1        | 130.0          | 2008-01-02 08:00 | 136.0      | 145.0
    #2          | 67890     | 2009      | 3                | 51222      | Hemoglobin | 13.5             | 2009-02-01 09:00 | 0       | NaN      | NaN            | NaN             | 12.0        | 18.0
    meta = build_labels(
        df=df,
        target_itemids=target_itemids,
        current_value_cols=current_value_cols,
        min_current_observed=args.min_current_observed,
    )

    if meta.empty:
        raise RuntimeError("No eligible rows were created.")

    # Filter the data to keep only these patients where we have at least one eligible lab row
    used_source_rows = meta["source_row"].drop_duplicates().to_numpy()
    X_current = X_current.loc[used_source_rows].reset_index(drop=True)
    source_to_embedding_row = {
        source_row: i for i, source_row in enumerate(used_source_rows)
    }
    meta["embedding_row"] = meta["source_row"].map(source_to_embedding_row)

    print(f"Embedding rows: {X_current.shape}; long label rows: {meta.shape}")

    if args.skip_embeddings:
        print("Skipping Lab-MAE embedding extraction.")
        out = meta.drop(columns=["embedding_row"])
    else:
        # Compute the embeddings for the current lab values using Lab-MAE, and pool them into a single row-level embedding z.
        imputer = load_imputer(args, dim=len(model_cols))
        print("Extracting and pooling Lab-MAE embeddings...")
        # Calculate embeddings and return a single embedding per row by pooling observed current lab embeddings.
        z = pooled_embeddings(imputer, X_current, args.batch_size)
        z_cols = [f"z_{i}" for i in range(z.shape[1])]
        z_df = pd.DataFrame(z, columns=z_cols)
        z_df["embedding_row"] = np.arange(len(z_df))
        out = meta.merge(z_df, on="embedding_row", how="left")
        out = out.drop(columns=["embedding_row"])

    out.to_csv(output_path, index=False)

    summary = (
        out.groupby(["lab_itemid", "lab_name"], dropna=False)
        .agg(
            rows=("ordered", "size"),
            ordered=("ordered", "sum"),
            abnormal=("abnormal", "sum"),
            ordered_rate=("ordered", "mean"),
            abnormal_rate_among_ordered=("abnormal", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"Wrote dataset: {output_path} shape={out.shape}")
    print(f"Wrote summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
