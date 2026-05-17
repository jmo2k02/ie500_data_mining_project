from __future__ import annotations

import argparse
import json
import os

from train import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multiclass flight departure-delay classifier."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "."),
        help="Local/Colab root or S3 URI, for example s3://data-mining.",
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Dataset path relative to --data-root. Supports .parquet and .csv.",
    )
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument(
        "--model",
        default="hist_gradient_boosting",
        choices=["dummy", "logistic_regression", "random_forest", "hist_gradient_boosting"],
    )
    parser.add_argument("--delay-column", default="DepDelayMinutes")
    parser.add_argument("--time-column", default="CRSDepDateTime_UTC")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        help="Optional MLflow experiment name. Omit to disable MLflow logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(
        TrainingConfig(
            data_root=args.data_root,
            input_path=args.input_path,
            output_dir=args.output_dir,
            model_name=args.model,
            delay_column=args.delay_column,
            time_column=args.time_column,
            sample_frac=args.sample_frac,
            tune=args.tune,
            n_iter=args.n_iter,
            cv_splits=args.cv_splits,
            mlflow_experiment=args.mlflow_experiment,
        )
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
