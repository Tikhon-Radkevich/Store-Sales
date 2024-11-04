import os

import pandas as pd
from darts import TimeSeries

from storesales.constants import SUBMISSIONS_PATH, EXTERNAL_SAMPLE_SUBMISSION_PATH


def save_submission(df: pd.DataFrame, file_name: str):
    df = df.set_index("id")

    submission_df = pd.read_csv(EXTERNAL_SAMPLE_SUBMISSION_PATH, index_col="id")
    submission_df["sales"] = df["yhat"]
    print(submission_df.isna().sum())

    file_path = os.path.join(SUBMISSIONS_PATH, file_name)
    submission_df.to_csv(file_path, index=True)


def train_test_split(series: dict[str, list[TimeSeries]], split_date: pd.Timestamp):
    train_series = {}

    for family, series_list in series.items():
        train_series[family] = [s.drop_after(split_date) for s in series_list]

    return train_series
