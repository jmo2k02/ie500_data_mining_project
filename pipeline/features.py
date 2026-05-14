from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from targets import DELAY_CLASS_ORDER


# list of Columns to Drop
COLS_TO_DROP = [
    "Information_time_UTC", # not relevant for a model
    "floor_informationtime_UTC",
    "Information_time",
    "floor_informationtime",
    "dep_hour", # aleady included in the CRSDepDateTime
    "ArrDelayMinutes", # target variable, should not be included as a feature, once the categorical target variable is created, the original delay minutes column should be dropped to avoid leakage   
    ]
# Datetime to convert to norminal:  min max scale them
DATIMES_TO_SCALE = [
    "CRSDepDateTime", # convert to the relative time of day, of the flight day and scale it to [0,1]
    "CRSArrDateTime" # convert to the relative time of day, of the flight day and scale it to [0,1]
    ] 
# list of Columns to OneHot Encode
ONE_HOT_COLS = [
    "Reporting_Airline", # the operating airline of the flight
    "Origin","Dest" # ORIGIN an DEST airports
    ]
# list of numerical columns to scale with min max scaler
NORMINAL_ENCODE =["2h_prev_flights_so_far_day","2h_prev_cum_count_so_far_day",
                "2h_prev_cum_cancelled_so_far_day",
                "DaysToNearestHoliday","Distance","LateAircraftDelay_prev_flight_delay_info",
                "WeatherDelay_prev_flight_delay_info","NASDelay_prev_flight_delay_info","CarrierDelay_prev_flight_delay_info"
                  ]
NORMINAL_ENCODE_WEATHER =["temp_DEP","temp_ARR", # temperature at departure and arrival airport 2h before the scheduled departure time
                          "prcp_DEP","prcp_ARR", # preciperation at departure and arrival airport 2h before the scheduled departure time
                          "wspd_DEP","wspd_ARR", # wind spead at departure and arrival airport 2h before the scheduled departure time
                          "rhum_DEP","rhum_ARR", # relative humitidy at departure and arrival airport 2h before the scheduled departure time
                          ]


NORMINAL_ENCODE_HIST = [
    "hist_airline_delay", "hist_airline_delay_7d", "hist_airline_delay_30d",
    "hist_origin_delay", "hist_origin_delay_7d", "hist_origin_delay_30d",
    "hist_dest_delay", "hist_dest_delay_7d", "hist_dest_delay_30d",
]

# NORMINAL_ENCODE_HIST = ["hd_airline_arr_all","hd_airline_arr_7d","hd_airline_arr_30d", # historical average arrival delay for the airline, calculated over all time, the last 7 days and the last 30 days
#                         "hd_airline_carrier_7d","hd_airline_carrier_30d", # historical average arrival delay, for the airline, which is confirmed carrier delay.
#                         "hd_Ori_dep_all","hd_Ori_dep_7d","hd_Ori_dep_30d", # historical average departure delay for the origin airport, calculated over all time, the last 7 days and the last 30 days
#                         "hd_Ori_nas_7d","hd_Ori_nas_30d", # historical average departure delay for the origin airport, which is confirmed NAS delay, calculated over the last 7 days and the last 30 days
#                         "hd_ground_crew_carrier_7d","hd_ground_crew_carrier_30d", # historical average departure delay for the origin airport, which is confirmed ground crew delay, calculated over the last 7 days and the last 30 days
#                         "hd_ground_crew_late_aircraft_3d","hd_ground_crew_late_aircraft_7d",
#                             "hd_ground_crew_late_aircraft_30d","hd_ground_crew_late_aircraft_all", # historical average departure delay for the origin airport, which is confirmed late aircraft delay, calculated over the last 7 days and the last 30 days
#                         "hd_Des_arr_all","hd_Des_arr_7d","hd_Des_arr_30d", # historical average arrival delay for the destination airport, calculated over all time, the last 7 days and the last 30 days
#                         "hd_Des_nas_7d","hd_Des_nas_30d",
#                         "hd_Des_in_air_7d","hd_Des_in_air_30d", 
#                         "hd_route_arr_7d","hd_route_arr_30d",
#                         "hd_route_nas_7d","hd_route_nas_30d",
#                         "hd_route_in_air_7d","hd_route_in_air_30d",
#                         "hd_route_airline_arr_7d","hd_route_airline_arr_30d",
#                         "hd_route_airline_carrier_7d","hd_route_airline_carrier_30d",
#                         "hd_route_airline_in_air_7d","hd_route_airline_in_air_30d",
# ]

NORMAL_ENCODE_LAG = [
    "origin_yesterday_delay", "origin_lastweek_delay",
    "dest_yesterday_delay", "dest_lastweek_delay",
    "airline_yesterday_delay", "airline_lastweek_delay",
    "global_yesterday_delay", "global_lastweek_delay",
]

# columsn to leave as is: 
LEAVE_AS_IS = ["Prev_flight_DelayMinutes","Expected_Tournaround_time",
               "DayOfYear","CRSTurnaroundTime",
               "DistanceGroup","Year","Month",
               "DayOfWeek","DayofMonth","CRSElapsedTime",
                 "Time_since_last_certified_record","Flights_before_today",
                 "total_Flights_scheduled_today","2h_prev_avg_delay_so_far_day"
               ]
# 
BOOLEAN_COLS = ["has_prev_flight",
                "IsHoliday",
                "FirstFlightRecord","is_Return_flight",
                "Same_day_previous_flight"]
# semi bolean (3 Values)
SEMI_BOOLEAN_COLS = ["Airplane_already_at_airport"] # 0 for not at the airport, 1 for already at the airport, 0.5 for unknown

TIME_COLUMN = "CRSDepDateTime_UTC"
TARGET_COLUMN = "delay_class"

# combine all the scaling columns for easier access in the fit_transform function
SCALING_COLS = DATIMES_TO_SCALE + NORMINAL_ENCODE + NORMINAL_ENCODE_WEATHER + NORMINAL_ENCODE_HIST + NORMAL_ENCODE_LAG

ALL_FEATURE_COLS = ONE_HOT_COLS + SCALING_COLS + LEAVE_AS_IS + BOOLEAN_COLS + SEMI_BOOLEAN_COLS
ALL_COLS_SET = set(ALL_FEATURE_COLS + [TARGET_COLUMN] + COLS_TO_DROP)

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

def check_columns(df: pd.DataFrame) -> None:
    all_cols = set(df.columns)
    recorded_cols = ALL_COLS_SET
    missing_cols = recorded_cols - all_cols
    extra_cols = all_cols - recorded_cols
    if missing_cols:
        print(f"Warning: The following columns are in the feature engineering lists but not in the dataset: {missing_cols}")
    if extra_cols:
        print(f"Info: The following columns are in the dataset but not recorded in the feature engineering lists: {extra_cols}")


# global
ONE_HOT_ENCODER =None # to be initialzied and fitted in the fit_transform function, to be used in the transform function, to ensure that the same encoding is applied to the training and test set.
SCALER = None # to be initialzied and fitted in the fit_transform function, to be used in the transform function, to ensure that the same scaling is applied to the training and test set.
LABEL_ENCODER = None # to be initialzied and fitted in the fit_transform function, to be used in the transform function, to ensure that the same encoding is applied to the training and test set.
def fit_transform(df: pd.DataFrame,scale:str="minmax") -> tuple[pd.DataFrame, pd.Series]:
    # print out all the cols that are not in one of the lists above, to check if we have missed any columns.
    all_cols = set(df.columns)
    recorded_cols = ALL_COLS_SET
    missing_cols = recorded_cols - all_cols
    extra_cols = all_cols - recorded_cols
    if missing_cols:
        print(f"Warning: The following columns are in the dataset but not recorded in the feature engineering lists: {missing_cols}")
    if extra_cols:
        print(f"Info: The following columns are in the dataset but not recorded in the feature engineering lists: {extra_cols}")


    # drop the columns that we will not use in our model
    df = df.drop(columns=COLS_TO_DROP)

    # One Hot Encode the One_hot_cols
    global ONE_HOT_ENCODER, SCALER
    ONE_HOT_ENCODER = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoded = ONE_HOT_ENCODER.fit_transform(df[ONE_HOT_COLS])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=ONE_HOT_ENCODER.get_feature_names_out(ONE_HOT_COLS), index=df.index)
    df = pd.concat([df.drop(columns=ONE_HOT_COLS), one_hot_encoded_df], axis=1)

    # con vert the Datetime columns to numeric values miuntes of the day 
    for col in DATIMES_TO_SCALE:
        df[col] = df[col].dt.hour * 60 + df[col].dt.minute

    # Min Max Scale the datetime columns
    if scale == "minmax":
        SCALER = MinMaxScaler()
    elif scale == "standard":
        SCALER = StandardScaler()
    else:
        raise ValueError(f"Invalid scale value: {scale}. Use 'minmax' or 'standard'.")
    df[SCALING_COLS] = SCALER.fit_transform(df[SCALING_COLS])

    # Label encode the target variable using the explicit order from DELAY_CLASS_ORDER
    global LABEL_ENCODER
    LABEL_ENCODER = {label: idx for idx, label in enumerate(DELAY_CLASS_ORDER)}
    target_values = df[TARGET_COLUMN]
    unknown_labels = set(target_values.dropna().unique()) - set(LABEL_ENCODER)
    if unknown_labels:
        raise ValueError(f"Unknown target labels found: {sorted(unknown_labels)}")
    df[TARGET_COLUMN] = target_values.map(LABEL_ENCODER).astype(int)
    # split into features and target
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y

def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # drop the columns that we will not use in our model
    df = df.drop(columns=COLS_TO_DROP)

    # One Hot Encode the One_hot_cols
    global ONE_HOT_ENCODER, SCALER
    one_hot_encoded = ONE_HOT_ENCODER.transform(df[ONE_HOT_COLS])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=ONE_HOT_ENCODER.get_feature_names_out(ONE_HOT_COLS), index=df.index)
    df = pd.concat([df.drop(columns=ONE_HOT_COLS), one_hot_encoded_df], axis=1)

    # con vert the Datetime columns to numeric values miuntes of the day 
    for col in DATIMES_TO_SCALE:
        df[col] = df[col].dt.hour * 60 + df[col].dt.minute

    # Min Max Scale the datetime columns
    df[SCALING_COLS] = SCALER.transform(df[SCALING_COLS])

    # Label encode the target variable using the same explicit order as in fit_transform
    global LABEL_ENCODER
    target_values = df[TARGET_COLUMN]
    unknown_labels = set(target_values.dropna().unique()) - set(LABEL_ENCODER)
    if unknown_labels:
        raise ValueError(f"Unknown target labels found during transform: {sorted(unknown_labels)}")
    df[TARGET_COLUMN] = target_values.map(LABEL_ENCODER).astype(int)

    # split into features and target
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y

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
