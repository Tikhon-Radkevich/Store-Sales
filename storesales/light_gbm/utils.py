import os
from tqdm import tqdm

import pandas as pd
from darts.models import LightGBMModel

from storesales.light_gbm.dataset import FamilyDataset
from storesales.constants import SUBMISSIONS_PATH, EXTERNAL_SAMPLE_SUBMISSION_PATH


def load_oil(external_oil_path: str) -> pd.DataFrame:
    oil_df = pd.read_csv(external_oil_path, parse_dates=["date"])
    oil_df.set_index("date", inplace=True)
    oil_df = oil_df.asfreq("D")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill()
    oil_df.dropna(inplace=True)
    return oil_df


def save_submission(df: pd.DataFrame, file_name: str):
    df = df.set_index("id")

    submission_df = pd.read_csv(EXTERNAL_SAMPLE_SUBMISSION_PATH, index_col="id")
    submission_df["sales"] = df["sales"]

    file_path = os.path.join(SUBMISSIONS_PATH, file_name)
    submission_df.to_csv(file_path, index=True)

    print(f"Submission saved to {file_path}")

    return submission_df


def make_submission_predictions(
    dataset: dict[str, FamilyDataset],
    models: dict[str, LightGBMModel],
    forecast_df: pd.DataFrame,
    horizon: int = 16,
) -> pd.DataFrame:
    """
    Make predictions for all families and stores, storing results in forecast_df.
    - forecast_df: DataFrame with multiindex (ds, family, store_nbr)
        - with target name: 'sales'
    """
    predictions = []

    for family, family_dataset in tqdm(dataset.items()):
        inputs = family_dataset.get_submission_inputs()
        pred_series = models[family].predict(n=horizon, show_warnings=False, **inputs)

        for store_nbr, pred in zip(family_dataset.stores, pred_series):
            pred_df = pred.pd_dataframe(copy=True)
            pred_df["family"] = family
            pred_df["store_nbr"] = store_nbr
            pred_df.set_index(["family", "store_nbr"], append=True, inplace=True)
            pred_df.index.names = ["ds", "family", "store_nbr"]
            predictions.append(pred_df)

        forecast_df.update(pd.concat(predictions)[["sales"]])

    return forecast_df


def make_submission_forecast_plot(
    dataset: dict[str, FamilyDataset],
    forecast: pd.DataFrame,
    family: str,
    i_series: int,
    drop_before_date: pd.Timestamp,
):
    store_nbr = dataset[family].stores[i_series]

    vals = dataset[family].series[i_series].drop_before(drop_before_date).pd_dataframe()

    con = (forecast["family"] == family) & (forecast["store_nbr"] == store_nbr)
    predict_vals = forecast[con][["ds", "sales"]]
    predict_vals.rename(columns={"ds": "date"}, inplace=True)
    predict_vals.set_index("date", inplace=True)

    title = f"{family} - Store {store_nbr}"
    pd.concat([vals, predict_vals], axis=1).plot(title=title)
