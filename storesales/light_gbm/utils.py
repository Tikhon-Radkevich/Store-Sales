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

    plt.figure(figsize=(10, n_top_features // 2))
    plt.barh(top_feature_names, top_feature_importances, color="skyblue")
    plt.xlabel("Feature Importance (Gain)")
    plt.title("Top 10 Feature Importance for LightGBM Model")
    plt.gca().invert_yaxis()
    plt.show()


def print_models_params(lgb_models: dict[str, LightGBMModel]):
    for family, lgb_model in lgb_models.items():
        print(f"{family}:")
        print(f"\t- max_depth: {lgb_model.model.max_depth}")
        print(f"\t- num_leaves: {lgb_model.model.num_leaves}")
        print(f"\t- learning_rate: {lgb_model.model.learning_rate}")
        print(f"\t- n_estimators: {lgb_model.model.n_estimators}")
        print(f"\t- top_rate: {lgb_model.model.top_rate}")
        print(f"\t- other_rate: {lgb_model.model.other_rate}")
        print(f"\t- max_bin: {lgb_model.model.max_bin}")
        print(f"\t- feature_fraction: {lgb_model.model.feature_fraction}")
        print(f"\t- max_cat_threshold: {lgb_model.model.max_cat_threshold}")
        print(f"\t- data_sample_strategy: {lgb_model.model.data_sample_strategy}\n\n")


def make_submission_predictions(
    dataset: dict[str, FamilyDataset],
    models: dict[str, LightGBMModel],
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

    forecast_df["sales"] = forecast_df["sales"].clip(lower=0)
    return forecast_df
