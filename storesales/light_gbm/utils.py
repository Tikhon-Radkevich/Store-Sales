from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import LightGBMModel

from storesales.light_gbm.dataset import FamilyDataset
from storesales.constants import START_SUBMISSION_DATE, EXTERNAL_TEST_PATH


def plot_feature_importance(lgb_model: LightGBMModel, n_top_features: int = 30):
    feature_importances = lgb_model.model.feature_importances_
    feature_names = lgb_model.lagged_feature_names

    top_indices = np.argsort(feature_importances)[-n_top_features:]
    top_feature_importances = feature_importances[top_indices]
    top_feature_names = np.array(feature_names)[top_indices]

    plt.figure(figsize=(10, 16))
    plt.barh(top_feature_names, top_feature_importances, color="skyblue")
    plt.xlabel("Feature Importance (Gain)")
    plt.title("Top 10 Feature Importance for LightGBM Model")
    plt.gca().invert_yaxis()
    plt.show()


def make_submission_predictions(
    dataset: dict[str, FamilyDataset],
    models: dict[str, LightGBMModel],
    # forecast_df: pd.DataFrame,
    horizon: int = 16,
) -> pd.DataFrame:
    """
    Make predictions for all families and stores, storing results in forecast_df.
    - forecast_df: DataFrame with multiindex (ds, family, store_nbr)
        - with target name: 'sales'
    """
    predictions = []

    forecast_df = pd.read_csv(EXTERNAL_TEST_PATH, parse_dates=["date"])
    forecast_df.set_index(["date", "family", "store_nbr"], inplace=True)
    forecast_df["sales"] = None

    for family, family_dataset in tqdm(dataset.items()):
        inputs = family_dataset.get_submission_inputs()
        pred_series = models[family].predict(n=horizon, show_warnings=False, **inputs)

        for store_nbr, pred in zip(family_dataset.stores, pred_series):
            pred_df = pred.pd_dataframe(copy=True)
            pred_df[["family", "store_nbr"]] = family, store_nbr
            pred_df.set_index(["family", "store_nbr"], append=True, inplace=True)
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

    vals = dataset[family].series[i_series].drop_before(drop_before_date)
    vals_df = vals.drop_after(pd.Timestamp(START_SUBMISSION_DATE)).pd_dataframe()

    con = (forecast["family"] == family) & (forecast["store_nbr"] == store_nbr)
    predict_vals = forecast[con][["ds", "sales"]]
    predict_vals.rename(columns={"ds": "date"}, inplace=True)
    predict_vals.set_index("date", inplace=True)

    title = f"{family} - Store {store_nbr}"
    pd.concat([vals_df, predict_vals], axis=1).plot(title=title)


def combine_baseline_losses(baseline_losses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # todo: why baseline losses processing is here??
    losses = []
    for model_name, loss_df in baseline_losses.items():
        index = pd.Index([model_name] * len(loss_df), name="model")
        losses.append(loss_df.set_index(index, append=True))

    baseline_losses_df = pd.concat(losses).sort_index()
    return baseline_losses_df
