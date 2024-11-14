import os

import pandas as pd

from config import ROOT_PATH


DATA_PATH = os.path.join(ROOT_PATH, "data")
EXTERNAL_DATA_PATH = os.path.join(DATA_PATH, "external")
LOSSES_DATA_PATH = os.path.join(DATA_PATH, "losses")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

DATASET_FILES = (
    "train.csv",
    "test.csv",
    "stores.csv",
    "transactions.csv",
    "oil.csv",
    "holidays_events.csv" "sample_submission.csv",
)
EXTERNAL_TRAIN_PATH = os.path.join(EXTERNAL_DATA_PATH, "train.csv")
EXTERNAL_TEST_PATH = os.path.join(EXTERNAL_DATA_PATH, "test.csv")
EXTERNAL_STORES_PATH = os.path.join(EXTERNAL_DATA_PATH, "stores.csv")
EXTERNAL_TRANSACTIONS_PATH = os.path.join(EXTERNAL_DATA_PATH, "transactions.csv")
EXTERNAL_OIL_PATH = os.path.join(EXTERNAL_DATA_PATH, "oil.csv")
EXTERNAL_HOLIDAYS_EVENTS_PATH = os.path.join(EXTERNAL_DATA_PATH, "holidays_events.csv")
EXTERNAL_SAMPLE_SUBMISSION_PATH = os.path.join(
    EXTERNAL_DATA_PATH, "sample_submission.csv"
)

REPORTS_PATH = os.path.join(ROOT_PATH, "reports")
SUBMISSIONS_PATH = os.path.join(DATA_PATH, "submissions")

HORIZON_INT = 16
HORIZON_STR = "16 days"

END_TRAIN_DATE = "2017-08-15"
START_SUBMISSION_DATE = "2017-08-16"
END_SUBMISSION_DATE = "2017-08-31"

START_VALIDATION_DATE = "2017-02-27"
START_TEST_DATE = "2017-05-10"
MIDDLE_TEST_DATE = "2017.06.20"

TEST_DATE_RANGE = pd.date_range(
    start=pd.Timestamp(START_TEST_DATE),
    end=pd.Timestamp(END_TRAIN_DATE) - pd.Timedelta(days=HORIZON_INT),
    freq="D",
)
VALIDATION_DATE_RANGE = pd.date_range(
    start=pd.Timestamp(START_VALIDATION_DATE),
    end=pd.Timestamp(START_TEST_DATE) - pd.Timedelta(days=HORIZON_INT + 1),
    freq="D",
)
