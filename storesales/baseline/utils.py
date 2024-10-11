from collections import defaultdict
from itertools import product
import random

from tqdm import tqdm
import numpy as np
import pandas as pd

import optuna

from storesales.baseline.loss import rmsle
from storesales.baseline.sales_predictor import SalesPredictor
from storesales.constants import (
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_SAMPLE_SUBMISSION_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_OIL_PATH,
    EXTERNAL_HOLIDAYS_EVENTS_PATH,
)


def make_time_series_split(
    df: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    test_size: int = 16,
):
    families = df["family"].unique()
    stores = df["store_nbr"].unique()
    total_pairs = len(families) * len(stores)
    dataset = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    test_period = pd.Timedelta(days=test_size)

    for family, store in tqdm(product(families, stores), total=total_pairs):
        store_family_df = df[(df["family"] == family) & (df["store_nbr"] == store)]

        for cutoff in cutoffs:
            start_test = cutoff.strftime("%Y-%m-%d")
            end_test = (cutoff + test_period).strftime("%Y-%m-%d")

            train_data = store_family_df.query(f" ds < '{start_test}' ")
            test_data = store_family_df.query(f" '{start_test}' <= ds < '{end_test}' ")

            dataset["train"][(store, family)].append(train_data)
            dataset["test"][(store, family)].append(test_data)

    return dataset


def run_study(df: pd.DataFrame, dataset, predictor: SalesPredictor):
    stores = df["store_nbr"].unique()

    for family_group in predictor.family_groups:
        store_family_groups = list(product(stores, family_group))
        n_choices = predictor.get_n_store_family_choices(family_group)

        sampled_store_family = random.sample(store_family_groups, n_choices)
        for i_sample, (store, family) in enumerate(sampled_store_family):
            print(f"\n\nFamily: {family} - Store: {store}")

            outer_rmsle = []
            for i_fold, outer_train in enumerate(dataset["train"][(store, family)]):
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: predictor.objective(trial, outer_train),
                    **predictor.optuna_optimize_kwargs,
                )

                test_loss = []
                for _store, _family in store_family_groups:
                    _outer_test = dataset["test"][(_store, _family)][i_fold]
                    _outer_train = dataset["train"][(_store, _family)][i_fold]

                    model = predictor.get_best_model(study.best_params)
                    model.fit(_outer_train)
                    forecast = model.predict(_outer_test)
                    y_pred = forecast["yhat"].values
                    y_true = _outer_test["y"].values
                    loss = rmsle(y_true, y_pred)
                    test_loss.append(loss)

                predictor.update_tune_loss_storage(family_group, test_loss, i_sample, i_fold)

                fold_test_loss = np.mean(test_loss)

                predictor.log_study(
                    family_group=family_group,
                    best_params=study.best_params,
                    loss=fold_test_loss,
                )
                outer_rmsle.append(fold_test_loss)
                print(
                    f"Outer: {predictor.render_model(study.best_params)} - RMSLE: {fold_test_loss}"
                )

            print(f"\nFamily: {family} - Store: {store} RMSLE: {np.mean(outer_rmsle)}")

        predictor.calc_and_log_mean_params(family_group)

    losses = [
        value["loss"]
        for _key, value in predictor.family_to_madel_params_storage.items()
    ]
    print(f"\n\nTotal RMSLE: {np.mean(losses)}")
    return predictor


def load_baseline_data():
    # load oil
    oil_df = pd.read_csv(EXTERNAL_OIL_PATH, parse_dates=["date"])
    oil_df.set_index("date", inplace=True)
    oil_df = oil_df.asfreq("D")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill()
    oil_df = oil_df.dropna()

    # load train
    original_train_df = pd.read_csv(EXTERNAL_TRAIN_PATH, parse_dates=["date"])
    original_train_df.sort_values(by=["date", "store_nbr", "family"], inplace=True)

    train_df = original_train_df[["date", "sales", "store_nbr", "family"]].copy()
    train_df.rename(columns={"date": "ds", "sales": "y"}, inplace=True)

    train_df = train_df.merge(oil_df, left_on="ds", right_index=True, how="left")
    train_df.dropna(inplace=True)

    train_df.sort_values(by="ds", inplace=True)

    # load test
    original_test_df = pd.read_csv(EXTERNAL_TEST_PATH, parse_dates=["date"])

    test_df = original_test_df[["date", "store_nbr", "family", "id"]].copy()
    test_df = test_df.merge(oil_df, left_on="date", right_index=True, how="left")
    test_df.rename(columns={"date": "ds"}, inplace=True)

    # load holidays
    holidays_df = pd.read_csv(EXTERNAL_HOLIDAYS_EVENTS_PATH, parse_dates=["date"])
    holidays_df = holidays_df[~holidays_df["transferred"]]
    holidays_df = holidays_df[["date", "description"]].rename(
        columns={"date": "ds", "description": "holiday"}
    )

    return train_df, test_df, holidays_df


def load_submission():
    return pd.read_csv(EXTERNAL_SAMPLE_SUBMISSION_PATH, index_col="id")
