from __future__ import annotations

import pandas as pd


DEFAULT_LEAKAGE_COLUMNS = [
    # Arrival delay values are only known after the flight has arrived.
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrivalDelayGroups",
    "ArrTime",

    # Actual flight duration / airborne time is not known two hours before departure.
    "ActualElapsedTime",
    "AirTime",

    # Delay reason columns are assigned after the delay happened, so they are target leakage.
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",

    # Departure delay values and actual departure movement are what we are trying to predict,
    # or are only known after the prediction cutoff.
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "DepTime",
    "TaxiOut",
    "WheelsOff",

    # Arrival movement timestamps and taxi-in are also future information at prediction time.
    "WheelsOn",
    "TaxiIn",

    # Cancellation/diversion information is generally not known at the two-hour cutoff and
    # changes the meaning of delay labels, so keep it out of the first training baseline.
    "Cancelled",
    "CancellationCode",
    "Diverted",
]


def build_feature_matrix(
    dataframe: pd.DataFrame,
    target_column: str,
    time_column: str,
    extra_drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return one-hot encoded features and target labels."""
    drop_columns = set(DEFAULT_LEAKAGE_COLUMNS)
    drop_columns.update({target_column, time_column, "FlightDate"})
    if extra_drop_columns:
        drop_columns.update(extra_drop_columns)

    existing_drop_columns = [column for column in drop_columns if column in dataframe.columns]
    y = dataframe[target_column].astype(str)
    x = dataframe.drop(columns=existing_drop_columns)

    datetime_columns = x.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    x = x.drop(columns=list(datetime_columns))
    x = pd.get_dummies(x, dummy_na=True)
    x = x.fillna(0)
    return x, y
