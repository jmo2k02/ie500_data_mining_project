# Flight Delay Prediction — Pipeline Reference

**Task:** Classify flight departure delay into four ordered categories two hours before the scheduled departure time.

**Target classes** (`targets.py`):

| Label | Delay range |
|---|---|
| `no_delay` | ≤ 15 min |
| `small_delay` | 15 – 60 min |
| `medium_delay` | 60 – 180 min |
| `large_delay` | > 180 min |

---

## Pipeline Stages

```
Raw CSVs (S3)
    └─ A: Loading & Cleaning
         └─ B: Feature Engineering
              └─ features/j_all_feature_engineered.parquet (S3)
                   └─ C: Target creation → Chronological split → Fit/transform → Train → Evaluate
```

---

## A — Data Loading & Cleaning (`A_loader.ipynb`, `loader.py`)

### Scope filters

Applied immediately after reading each monthly CSV to keep RAM low:

- **Airlines** (`AIRLINE_LIMIT_LIST`): AS, AA, DL, F9, B6, WN, UA (7 major carriers; regional airlines and merged/defunct carriers excluded).
- **Airports** (`AIRPORT_LIMIT_LIST`): 35 major US airports — a flight is kept if either its origin or destination is in this list.
- **Columns** (`KEEP_COLS`): Only the 28 necessary columns are read from CSV; all others are discarded at parse time.

### Anomaly removal

- Flights with a missing actual arrival time that are neither cancelled nor diverted are dropped (aircraft never reached destination).
- Time columns (`CRSDepTime`, `DepTime`, `CRSArrTime`, `ArrTime`) are normalised: `2400` is replaced with `0000`, then parsed to `datetime.time`; NaT is set for cancelled/diverted legs.

### Timezone conversion

- The airport reference file (`data/external/airports_with_runway_info.csv`) is joined twice — once on `Origin`, once on `Dest` — to attach the IANA timezone string (`TZ_Origin`, `TZ_Dest`) to every flight.
- All four time columns are localised to their respective airport timezone and converted to UTC:
  - `CRSDepDateTime_UTC`, `CRSArrDateTime_UTC` — scheduled times
  - `DepDateTime_UTC`, `ArrDateTime_UTC` — actual times
- Overnight flights are detected by comparing actual vs. scheduled times and a `+1 day` offset is applied where necessary.
- CRS arrival times that are earlier than CRS departure times in UTC (red-eye routes) are shifted forward by one day.

### Information-time columns

The **information time** represents what is knowable 2 hours before scheduled departure:

| Column | Description |
|---|---|
| `Information_time_UTC` | `CRSDepDateTime_UTC − 2 h` |
| `floor_informationtime_UTC` | Floored to the nearest full hour (used for merging hourly weather data) |
| `Information_time` | Same in local time |
| `floor_informationtime` | Local floored hour |

### Output

Saved to `data/interim/1_year_data_new.parquet` (and analogous multi-year variants) on S3.

---

## B — Feature Engineering (`B_feature_engineering.ipynb`)

All features are constructed so that **no information past the 2-hour pre-departure cutoff leaks into the model**.

### 1. Real-time airport delay statistics

Captures the developing delay situation at the departure airport on the day of the flight, as seen at the information time.

**Process:**
1. Sort actual departure events by `(Origin, FlightDate, DepDateTime)`.
2. Compute cumulative statistics within each `(Origin, FlightDate)` group:
   - `cum_delay` — cumulative departure delay minutes of all flights so far.
   - `cum_count` — cumulative count of actually-departed flights.
   - `cum_cancelled_flights` — cumulative count of cancellations.
   - `avg_delay` — rolling average departure delay (`cum_delay / cum_count`).
3. Aggregate to hourly granularity: keep only the last record per `(Origin, FlightDate, ceil-hour)`, which holds the final cumulative state for that hour.
4. Compute within-hour statistics:
   - `information_hour_average_delay` — mean delay of flights that departed in that specific hour.
   - `information_hour_flight_count` — number of flights that departed in that hour.
5. Merge back onto each subject flight via `floor_informationtime_UTC`, so each flight sees the state of its origin airport at the last complete hour before the information time.

**Resulting features:**

| Feature | Meaning |
|---|---|
| `2h_prev_avg_delay_so_far_day` | Average departure delay at origin airport so far that day |
| `2h_prev_cum_count_so_far_day` | Total departed flights at origin so far that day |
| `2h_prev_cum_cancelled_so_far_day` | Total cancellations at origin so far that day |
| `2h_prev_flights_so_far_day` | Flights departed in the last full hour at origin |

### 2. Turnaround time & previous-flight chain

Identifies each aircraft's immediately preceding flight and derives time/status features.

**Process:**
1. Sort the dataset by `(Tail_Number, CRSDepDateTime_UTC)`.
2. Shift all relevant columns by one row to get the "previous row" as a candidate previous flight.
3. Validate the candidate with three checks:
   - Tail number and airline match the previous row (confirms it is the same aircraft, or marks `FirstFlightRecord = 1` otherwise).
   - The previous flight was neither cancelled nor diverted.
   - The previous flight's destination matches the current flight's origin.
4. Only rows passing all three checks are considered a confirmed previous flight (`has_prev_flight = 1`).

**Delay propagation at information time:**

Given the 2-hour pre-departure information time, the pipeline determines the known status of the previous flight:

| Case | Feature value |
|---|---|
| Previous flight has not yet reached its scheduled departure | `Prev_flight_DelayMinutes = 0` |
| Previous flight has departed → use recorded departure delay | `Prev_flight_DelayMinutes = PreviousFlightDepDelayMinutes` |
| Previous flight should have departed but has not → estimate delay from current time | `Prev_flight_DelayMinutes = (Information_time_UTC − CRSPreviousFlightDepDateTime_UTC)` |
| Previous flight has already arrived | `Prev_flight_DelayMinutes = PreviousFlightArrDelayMinutes` |
| Previous flight should have arrived but has not → add running lateness | Time difference from scheduled arrival |

**`Airplane_already_at_airport`:** `1.0` if the previous flight has arrived, `0.0` if not, `0.5` if unknown (no previous flight record).

**Turnaround features:**

| Feature | Description |
|---|---|
| `CRSTurnaroundTime` | Scheduled turnaround in minutes (`CRSDepDateTime_UTC − CRSPreviousFlightArrDateTime_UTC`); filled with mean where unknown |
| `Expected_Tournaround_time` | `CRSTurnaroundTime − Prev_flight_DelayMinutes` — expected shortfall in ground time |
| `Flights_before_today` | How many legs the same aircraft has already flown today |
| `total_Flights_scheduled_today` | Total legs scheduled for the aircraft on this date |
| `has_prev_flight` | Binary flag for confirmed previous flight |
| `FirstFlightRecord` | Binary flag for first-ever record of this tail number |
| `Same_day_previous_flight` | 1 if the previous flight was on the same calendar day |
| `is_Return_flight` | 1 if Origin/Dest are exactly swapped relative to the previous leg |

**Previous-flight delay reasons** (only populated when `Airplane_already_at_airport = 1`):

- `LateAircraftDelay_prev_flight_delay_info`
- `WeatherDelay_prev_flight_delay_info`
- `NASDelay_prev_flight_delay_info`
- `CarrierDelay_prev_flight_delay_info`

### 3. Holiday features

Uses the `holidays` Python library for official US federal holidays; Christmas Eve (Dec 24) and New Year's Eve (Dec 31) are added manually.

| Feature | Description |
|---|---|
| `IsHoliday` | Binary flag — flight date is a US public holiday |
| `DaysToNearestHoliday` | Minimum calendar distance to any holiday in the dataset's year range |

### 4. Weather data (`meteostat`)

Hourly weather is fetched for each airport in `AIRPORT_LIMIT_LIST` for the date range spanned by that airport's flights. If the primary airport station has gaps, up to three nearby stations within 50 km are used to fill missing hours. Remaining gaps are forward-filled (max 2 steps); airports with > 50 % missing after imputation are filled with 0.

Merged onto each flight twice — once for the departure airport, once for the destination — at the `floor_informationtime_UTC` key:

| Feature suffix | Parameter |
|---|---|
| `temp_DEP` / `temp_ARR` | Air temperature (°C) |
| `prcp_DEP` / `prcp_ARR` | Precipitation (mm/h) |
| `wspd_DEP` / `wspd_ARR` | Wind speed (km/h) |
| `rhum_DEP` / `rhum_ARR` | Relative humidity (%) |

Flights where either the departure or arrival weather merge produces no match are dropped.

### 5. Departure hour

`dep_hour = CRSDepDateTime.dt.hour` — an integer 0–23 capturing time-of-day delay patterns.

### 6. Rolling historical delay features

Computed from daily mean `ArrDelayMinutes`, grouped by airline, origin, and destination. `shift(1)` is applied before every rolling calculation to prevent the current day from leaking into its own historical average.

| Feature | Window |
|---|---|
| `hist_airline_delay` | Expanding mean (all prior days) |
| `hist_airline_delay_7d` | 7-day rolling mean |
| `hist_airline_delay_30d` | 30-day rolling mean |
| `hist_origin_delay` | Expanding mean per origin airport |
| `hist_origin_delay_7d` | 7-day rolling mean per origin airport |
| `hist_origin_delay_30d` | 30-day rolling mean per origin airport |
| `hist_dest_delay` | Expanding mean per destination airport |
| `hist_dest_delay_7d` | 7-day rolling mean per destination airport |
| `hist_dest_delay_30d` | 30-day rolling mean per destination airport |

### 7. Lag features

Yesterday (lag-1) and last-week (lag-7) daily mean arrival delay, motivated by autocorrelation analysis (lag-1 ≈ 0.53, lag-7 ≈ 0.21).

Computed per origin airport, destination airport, airline, and globally:

| Feature | Description |
|---|---|
| `origin_yesterday_delay` | Mean ArrDelayMinutes at origin airport yesterday |
| `origin_lastweek_delay` | Mean ArrDelayMinutes at origin airport same weekday last week |
| `dest_yesterday_delay` / `dest_lastweek_delay` | Same for destination airport |
| `airline_yesterday_delay` / `airline_lastweek_delay` | Same per airline |
| `global_yesterday_delay` / `global_lastweek_delay` | System-wide daily mean |

All lag/rolling features are zero-filled for the first days of the dataset where no history exists.

### Final filtering before save

- Flights with missing `CRSElapsedTime` are removed.
- Only flights where **both** origin and destination are in `AIRPORT_LIMIT_LIST` are kept.
- Cancelled and diverted flights are removed.
- The `year < min_year` warm-up rows (used only for historical feature bootstrapping) are dropped.

**Output:** `data/features/feature_engineered.parquet` (or `j_all_feature_engineered.parquet` for the joined multi-year file) on S3.

---

## C — Feature Transformation (`features.py`)

Applied inside the training pipeline via `fit_transform` (train set) and `transform` (val/test sets), ensuring no statistics from val/test leak into the scaler or encoder.

### Columns dropped (`COLS_TO_DROP`)

These are either redundant, index artifacts, or intermediate computation columns not needed by the model:

`Information_time_UTC`, `floor_informationtime_UTC`, `Information_time`, `floor_informationtime`, `FlightID_prev_flight_delay_info`, `dep_hour` (already encoded in `CRSDepDateTime`), `ArrDelayMinutes` (target source, replaced by `delay_class`).

### One-hot encoding (`ONE_HOT_COLS`)

`Reporting_Airline`, `Origin`, `Dest` — produces sparse 0/1 columns; unknown categories at inference time are silently ignored (`handle_unknown="ignore"`).

### Datetime → minutes-of-day, then scaled

`CRSDepDateTime` and `CRSArrDateTime` are converted to integer minutes since midnight (`hour × 60 + minute`) before being passed to the scaler.

### Min-Max (or Standard) scaling (`SCALING_COLS`)

All of the following groups are fit on the training set and applied identically to val/test:

- Datetime columns (converted to minutes-of-day above)
- `NORMINAL_ENCODE`: flight-volume and delay counts/durations (`2h_prev_*`, `DaysToNearestHoliday`, `Distance`, previous-flight delay reasons)
- `NORMINAL_ENCODE_WEATHER`: all 8 weather parameters
- `NORMINAL_ENCODE_HIST`: 9 rolling historical delay features
- `NORMAL_ENCODE_LAG`: 8 lag features

### Left as-is (`LEAVE_AS_IS`)

Integer/float columns that are already on a meaningful scale or are categorical integers:
`Prev_flight_DelayMinutes`, `Expected_Tournaround_time`, `DayOfYear`, `CRSTurnaroundTime`, `DistanceGroup`, `Year`, `Month`, `DayOfWeek`, `DayofMonth`, `CRSElapsedTime`, `Time_since_last_certified_record`, `Flights_before_today`, `total_Flights_scheduled_today`, `2h_prev_avg_delay_so_far_day`.

### Boolean columns

`has_prev_flight`, `IsHoliday`, `FirstFlightRecord`, `is_Return_flight`, `Same_day_previous_flight` — kept as 0/1 integers.

`Airplane_already_at_airport` — semi-boolean with three values (0, 0.5, 1).

### Target encoding (`delay_class`)

Ordinal label encoding using the fixed order `no_delay=0, small_delay=1, medium_delay=2, large_delay=3`. The mapping is applied identically to all splits using the same dictionary fitted on the training set.

**Final feature matrix:** 133 columns after one-hot expansion.

---

## D — Chronological Train / Val / Test Split (`split.py`)

Rows are sorted by `CRSDepDateTime_UTC` and sliced by position (not randomly):

| Split | Share | Approx. size (4 M rows) |
|---|---|---|
| Train | 70 % | ~2.83 M rows |
| Validation | 15 % | ~606 K rows |
| Test | 15 % | ~606 K rows |

`CRSDepDateTime_UTC` is dropped from all splits after splitting to prevent it being used as a model feature.

**Why chronological:** Flight delay patterns exhibit temporal autocorrelation. A random split would allow the model to see future information in training, inflating performance estimates.

---

## E — Target Creation (`targets.py`)

`add_delay_class_target` cuts `ArrDelayMinutes` (or `DepDelayMinutes`) into the four bins using `pd.cut` with `right=True`. Rows where the cut produces NaN (flights with missing delay) are dropped.

Class balance in the ~4 M-row dataset:

| Class | Share |
|---|---|
| `no_delay` | ~80.1 % |
| `small_delay` | ~13.9 % |
| `medium_delay` | ~5.1 % |
| `large_delay` | ~0.85 % |

---

## F — Training (`train.py`, `C_training.ipynb`)

### Class weights

The strong class imbalance is addressed via `sample_weight` passed to `model.fit`. Two strategies are available:

- **`"balanced"`** — `sklearn.utils.class_weight.compute_class_weight` derives weights inversely proportional to class frequency.
- **Manual dict** (default in `run_training` when `weights=None`): `{0: 0.3, 1: 0.7, 2: 1.0, 3: 1.5}` — penalises errors on rare classes more.

`compute_sample_weight` maps the per-class weights onto individual training rows.

### Available models (`models.py`)

| Key | Estimator | Notes |
|---|---|---|
| `dummy` | `DummyClassifier(strategy="most_frequent")` | Always predicts `no_delay`; lower bound |
| `Baseline` | `BaselineModel` | Bins `2h_prev_avg_delay_so_far_day` into the four classes using the same thresholds as the target |
| `logistic_regression` | `LogisticRegression(max_iter=500, class_weight="balanced")` | Fast linear baseline |
| `logistic_regression_pipeline` | `Pipeline([StandardScaler, LogisticRegression])` | Same with in-pipeline scaling |
| `random_forest` | `RandomForestClassifier(n_estimators=200, min_samples_leaf=10, class_weight="balanced_subsample")` | |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier(lr=0.08, max_iter=200)` | Handles missing values natively |
| `xgboost` | `XGBClassifier(lr=0.1, n_estimators=100, max_depth=8, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss")` | **Current best model** |
| `naive_bayes` | `MultinomialNB` | |
| `svc` | `SVC(class_weight="balanced")` | |
| Neural network | `keras.Sequential` (512→Dropout→128→Dropout→32→4 softmax) | Adam lr=1e-3, `sparse_categorical_crossentropy`, optional |

### Outputs persisted (`_persist_outputs`)

| File | Contents |
|---|---|
| `{model_name}.joblib` | Serialised trained model |
| `metrics.json` | Validation and test metric dictionaries |
| `features.json` | Ordered list of feature column names |
| `classification_report_test.csv` | Per-class precision / recall / F1 on the test set |

Optional MLflow logging is triggered when `mlflow_experiment` is set in `TrainingConfig`.

---

## G — Hyperparameter Tuning (`D_hyperparameter_tuning.ipynb`, `train.py._tune_model`)

Enabled by setting `tune=True` in `TrainingConfig`.

- **Search method:** `RandomizedSearchCV` with `n_iter` random draws (default 20).
- **Cross-validation:** `TimeSeriesSplit(n_splits=3)` — respects temporal ordering within the training fold.
- **Scoring metric:** `f1_macro`.
- **Parallelism:** `n_jobs=-1`.

Search spaces per model (`default_param_distributions` in `models.py`):

| Model | Parameters searched |
|---|---|
| `logistic_regression` | `C`: [0.01, 0.1, 1.0, 10.0] |
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| `hist_gradient_boosting` | `learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization` |
| `xgboost` | `learning_rate`, `n_estimators`, `max_depth` |

The best estimator from the search is used for all downstream evaluation and persistence steps.

---

## H — Evaluation (`evaluate.py`)

### Metrics reported

| Metric | Function | Description |
|---|---|---|
| `accuracy` | `accuracy_score` | Fraction of correct predictions |
| `balanced_accuracy` | `balanced_accuracy_score` | Mean recall per class — corrects for imbalance |
| `macro_f1` | `f1_score(average="macro")` | Unweighted mean F1 across all four classes — **primary metric** |
| `weighted_f1` | `f1_score(average="weighted")` | F1 weighted by class support |

### Per-class report

`classification_report_frame` returns the full sklearn classification report as a DataFrame with precision, recall, and F1 for each of the four delay classes plus macro/weighted averages.

### Confusion matrix

`plotConfusionMatrix` renders a seaborn heatmap of the raw count confusion matrix with class labels on both axes.

### Feature importances (`models.py`)

`get_feature_importances` extracts importances in a model-agnostic way:

- **Tree models** (Random Forest, HGBT, XGBoost): `model.feature_importances_`
- **Logistic Regression**: sum of absolute coefficient values across classes, normalised
- **Neural network**: sum of absolute first-layer weights per input neuron, normalised

`plot_feature_importances` renders a horizontal bar chart of the top-N features.

### Observed results (XGBoost, full dataset)

| Split | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Test (with class weights) | 0.8626 | 0.5180 | 0.5697 | 0.8535 |

Per-class on test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `no_delay` | 0.910 | 0.951 | 0.930 | 511 678 |
| `small_delay` | 0.465 | 0.377 | 0.416 | 66 912 |
| `medium_delay` | 0.671 | 0.423 | 0.519 | 23 028 |
| `large_delay` | 0.582 | 0.322 | 0.414 | 4 801 |

---

## Data Flow Summary

```
S3: data/raw/*.csv
  │
  │  A_loader.ipynb
  │  • Filter to 7 airlines, 35 airports, 28 columns
  │  • Remove anomalous flights
  │  • Parse and convert times to UTC + local
  │  • Join airport timezone info
  │  • Compute information_time (CRSDep − 2h)
  ▼
S3: data/interim/1_year_data_new.parquet
  │
  │  B_feature_engineering.ipynb
  │  • Real-time airport delay stats (cum delay, count, cancellations)
  │  • Turnaround time & previous-flight chain
  │  • Previous-flight delay propagation at information time
  │  • Holiday flags & DaysToNearestHoliday
  │  • Hourly weather (meteostat) for departure + arrival airports
  │  • dep_hour
  │  • Rolling historical delay (7d / 30d / expanding) × 3 groupings
  │  • Lag-1 / Lag-7 features × 4 groupings
  │  • Drop cancelled, diverted, out-of-scope rows
  ▼
S3: data/features/j_all_feature_engineered.parquet
  │
  │  C_training.ipynb / train.py
  │  • add_delay_class_target → cut ArrDelayMinutes into 4 bins
  │  • chronological_train_val_test_split (70 / 15 / 15)
  │  • fit_transform on train: drop cols, OHE, MinMax scale, label encode target
  │  • transform on val + test (same fitted encoder/scaler)
  │  • Compute sample weights for class imbalance
  │  • model.fit(X_train, y_train, sample_weight=...)
  │  • evaluate_classifier on val and test
  │  • Persist model, metrics, features, classification report
  ▼
outputs/{model_name}/
  ├── {model_name}.joblib
  ├── metrics.json
  ├── features.json
  └── classification_report_test.csv
```
