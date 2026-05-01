# %% [markdown]
# # Flight Delay Training Pipeline Walkthrough
#
# This file is a VS Code-friendly alternative to the `.ipynb` notebook.
# VS Code can run each `# %%` block as a notebook cell, but the file still opens
# as normal Python text if the Jupyter webview is broken.
#
# Goal: classify departure delay into four interval classes two hours before departure:
#
# - `no_delay`: delay <= 15 minutes
# - `small_delay`: 15 < delay <= 30 minutes
# - `medium_delay`: 30 < delay <= 60 minutes
# - `large_delay`: delay > 60 minutes

# %% [markdown]
# ## 1. Setup
#
# Run this file from the `pipeline/` folder.
#
# In Colab or a fresh environment, install dependencies once:
#
# ```python
# %pip install pandas numpy scikit-learn pyarrow s3fs joblib
# ```

# %%
from pathlib import Path
import os
import sys
import pandas as pd

# Make local imports work when the file is opened from another folder.
PIPELINE_DIR = Path.cwd()
if not (PIPELINE_DIR / "loader.py").exists():
    PIPELINE_DIR = Path.cwd() / "pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from loader import LoaderStorage
from targets import add_delay_class_target, DELAY_CLASS_ORDER
from split import chronological_train_val_test_split
from features import build_feature_matrix
from models import make_model
from evaluate import evaluate_classifier, classification_report_frame
from train import TrainingConfig, run_training

# %% [markdown]
# ## 2. Configuration
#
# Only change this cell for normal usage.
#
# Examples:
#
# - Colab/Drive: `DATA_ROOT = "/content/drive/MyDrive/Datamining"`
# - S3: `DATA_ROOT = "s3://data-mining"`
# - Local folder: `DATA_ROOT = "."`

# %%
# Change these paths to your actual dataset location.
DATA_ROOT = os.environ.get("DATA_ROOT", ".")
INPUT_PATH = "Data_for_Model/preprocessed.parquet"

# Column names used by the current pipeline.
DELAY_COLUMN = "DepDelayMinutes"
TIME_COLUMN = "CRSDepDateTime_UTC"
TARGET_COLUMN = "delay_class"

# Use a small sample while learning/debugging. Set to 1.0 for the final run.
SAMPLE_FRAC = 0.05

# Good first choices: "dummy", "logistic_regression", "random_forest", "hist_gradient_boosting".
MODEL_NAME = "hist_gradient_boosting"

OUTPUT_DIR = "outputs/training_notebook"

# %% [markdown]
# ## 3. Load The Data
#
# `LoaderStorage` hides whether the file is loaded from local disk, Google Drive, or S3.
# The rest of the pipeline can use the same code for all three.

# %%
storage = LoaderStorage(DATA_ROOT)

if INPUT_PATH.endswith(".parquet"):
    df = storage.read_parquet(INPUT_PATH)
elif INPUT_PATH.endswith(".csv"):
    df = storage.read_csv(INPUT_PATH)
else:
    raise ValueError("Use a .parquet or .csv dataset.")

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=42)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
df.head()

# %% [markdown]
# ## 4. Create The Target Classes
#
# The raw delay in minutes is converted into the four interval classes.
# After this step, the model predicts `delay_class`, not the exact number of minutes.

# %%
df = add_delay_class_target(
    df,
    delay_column=DELAY_COLUMN,
    target_column=TARGET_COLUMN,
)

# Show the class balance. This is important because large delays are usually rare.
class_distribution = df[TARGET_COLUMN].value_counts(normalize=True).reindex(
    DELAY_CLASS_ORDER
)
class_distribution.to_frame("share")

# %% [markdown]
# ## 5. Chronological Train / Validation / Test Split
#
# For a forecasting-like task, we should not randomly mix old and future flights.
# The model trains on older flights and is evaluated on later flights.
#
# - Train: first 70% of time
# - Validation: next 15%
# - Test: final 15%

# %%
train_df, val_df, test_df = chronological_train_val_test_split(
    df,
    time_column=TIME_COLUMN,
)

print(f"Train rows:      {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")
print(f"Test rows:       {len(test_df):,}")

pd.DataFrame(
    {
        "split": ["train", "validation", "test"],
        "start": [
            train_df[TIME_COLUMN].min(),
            val_df[TIME_COLUMN].min(),
            test_df[TIME_COLUMN].min(),
        ],
        "end": [
            train_df[TIME_COLUMN].max(),
            val_df[TIME_COLUMN].max(),
            test_df[TIME_COLUMN].max(),
        ],
        "rows": [len(train_df), len(val_df), len(test_df)],
    }
)

# %% [markdown]
# ## 6. Build Features
#
# This step separates `X` and `y`:
#
# - `X`: the input columns the model is allowed to use
# - `y`: the delay class the model should learn to predict
#
# The helper drops obvious leakage columns like actual delay, actual departure time,
# actual arrival time, etc.

# %%
X_train, y_train = build_feature_matrix(train_df, TARGET_COLUMN, TIME_COLUMN)
X_val, y_val = build_feature_matrix(val_df, TARGET_COLUMN, TIME_COLUMN)
X_test, y_test = build_feature_matrix(test_df, TARGET_COLUMN, TIME_COLUMN)

# One-hot encoding can create different columns per split. Reindex keeps them aligned.
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"Number of model features: {len(X_train.columns):,}")
X_train.head()

# %% [markdown]
# ## 7. Train A Simple Baseline
#
# Always train a baseline first. If a complex model does not beat this, something is
# wrong or the features are weak.

# %%
baseline = make_model("dummy")
baseline.fit(X_train, y_train)

baseline_metrics = evaluate_classifier(baseline, X_val, y_val)
pd.Series(baseline_metrics, name="baseline_validation_metrics")

# %% [markdown]
# ## 8. Train The Selected Model
#
# Now train the model chosen in the configuration cell. For large datasets, start with
# a small sample and only later set `SAMPLE_FRAC = 1.0`.

# %%
model = make_model(MODEL_NAME)
model.fit(X_train, y_train)

val_metrics = evaluate_classifier(model, X_val, y_val)
pd.Series(val_metrics, name=f"{MODEL_NAME}_validation_metrics")

# %% [markdown]
# ## 9. Final Test Evaluation
#
# Only use the test set after choosing a model. This gives the honest final estimate
# for the report.

# %%
test_metrics = evaluate_classifier(model, X_test, y_test)
pd.Series(test_metrics, name=f"{MODEL_NAME}_test_metrics")

# %%
# Per-class report. Look especially at recall for medium_delay and large_delay.
classification_report_frame(model, X_test, y_test)

# %% [markdown]
# ## 10. Run The Whole Pipeline In One Cell
#
# The cells above show each step. For actual experiments, use `run_training(...)`,
# which runs the same steps and saves outputs automatically.

# %%
config = TrainingConfig(
    data_root=DATA_ROOT,
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
    model_name=MODEL_NAME,
    delay_column=DELAY_COLUMN,
    target_column=TARGET_COLUMN,
    time_column=TIME_COLUMN,
    sample_frac=SAMPLE_FRAC,
    tune=False,
)

metrics = run_training(config)
pd.Series(metrics, name="pipeline_metrics")

# %% [markdown]
# ## 11. Optional: Hyperparameter Tuning
#
# Tuning tries multiple parameter combinations with `TimeSeriesSplit`. This can take a
# long time. Use it only after the simple training run works.

# %%
# Uncomment this block when you are ready for a slower tuning run.
# tuned_config = TrainingConfig(
#     data_root=DATA_ROOT,
#     input_path=INPUT_PATH,
#     output_dir="outputs/training_notebook_tuned",
#     model_name=MODEL_NAME,
#     delay_column=DELAY_COLUMN,
#     target_column=TARGET_COLUMN,
#     time_column=TIME_COLUMN,
#     sample_frac=SAMPLE_FRAC,
#     tune=True,
#     n_iter=20,
#     cv_splits=3,
# )
# tuned_metrics = run_training(tuned_config)
# pd.Series(tuned_metrics, name="tuned_pipeline_metrics")

# %% [markdown]
# ## 12. What To Report
#
# For the university report, include:
#
# - target class definitions
# - chronological split dates
# - class distribution
# - baseline metrics
# - final model metrics
# - per-class precision and recall
# - error analysis for `medium_delay` and `large_delay`
#
# Accuracy alone is not enough because most flights are probably `no_delay`.
