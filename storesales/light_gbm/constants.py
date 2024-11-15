import pandas as pd


TRAINING_DATA_THRESHOLD_TIMESTAMP = pd.Timestamp("2015-05-01")
ROLLS_THRESHOLD_TIMESTAMP = pd.Timestamp("2013-05-01")
START_TARGET_SERIES_TIMESTAMP = ROLLS_THRESHOLD_TIMESTAMP - pd.Timedelta(days=30)

FEATURES_TO_ROLL = ["sales", "onpromotion", "dcoilwtico"]

STATIC_COLS = ["city", "state", "type", "cluster"]
CAT_STATIC_COVS = ["city", "state", "type", "cluster", "store_nbr"]
CAT_FUTURE_COVS = []
