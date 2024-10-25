import os

from config import ROOT_PATH


DATA_PATH = os.path.join(ROOT_PATH, "data")
EXTERNAL_DATA_PATH = os.path.join(DATA_PATH, "external")

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

START_TEST_DATE = "2017-08-16"
END_TEST_DATE = "2017-08-31"
TRAIN_TEST_SPLIT_DATE = "2017-05-10"
