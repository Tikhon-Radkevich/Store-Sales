import os

import pandas as pd

from storesales.constants import EXTERNAL_SAMPLE_SUBMISSION_PATH, SUBMISSIONS_PATH


def load_oil(external_oil_path: str) -> pd.DataFrame:
    oil_df = pd.read_csv(external_oil_path, parse_dates=["date"])
    oil_df.set_index("date", inplace=True)
    oil_df = oil_df.asfreq("D")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill()
    oil_df.dropna(inplace=True)
    return oil_df


def load_stores(
    external_stores_path: str, factorize_cols: list[str] | None = None
) -> pd.DataFrame:
    stores_df = pd.read_csv(external_stores_path)

    if factorize_cols is None:
        return stores_df

    for col in factorize_cols:
        stores_df[col], _ = pd.factorize(stores_df[col], sort=True)

    return stores_df


def save_submission(df: pd.DataFrame, file_name: str):
    df = df.set_index("id")

    submission_df = pd.read_csv(EXTERNAL_SAMPLE_SUBMISSION_PATH, index_col="id")
    submission_df["sales"] = df["sales"]

    file_path = os.path.join(SUBMISSIONS_PATH, file_name)
    submission_df.to_csv(file_path, index=True)

    print(f"Submission saved to {file_path}")

    return submission_df
