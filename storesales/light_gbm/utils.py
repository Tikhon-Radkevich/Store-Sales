import os

import pandas as pd

from storesales.constants import SUBMISSIONS_PATH, EXTERNAL_SAMPLE_SUBMISSION_PATH


def save_submission(df: pd.DataFrame, file_name: str):
    df = df.set_index("id")

    submission_df = pd.read_csv(EXTERNAL_SAMPLE_SUBMISSION_PATH, index_col="id")
    submission_df["sales"] = df["yhat"]

    file_path = os.path.join(SUBMISSIONS_PATH, file_name)
    submission_df.to_csv(file_path, index=True)

    print(f"Submission saved to {file_path}")

    return submission_df
