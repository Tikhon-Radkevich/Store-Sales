import pandas as pd


TRAINING_DATA_THRESHOLD_TIMESTAMP = pd.Timestamp("2017-04-01")
ROLLS_THRESHOLD_TIMESTAMP = pd.Timestamp("2015-02-01")
START_TARGET_SERIES_TIMESTAMP = ROLLS_THRESHOLD_TIMESTAMP - pd.Timedelta(days=10)

FEATURES_TO_ROLL = ["sales", "onpromotion", "dcoilwtico"]
