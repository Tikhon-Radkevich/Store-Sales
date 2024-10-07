from itertools import product
import random

import numpy as np
import pandas as pd

import optuna

from storesales.baseline.sales_predictor import SalesPredictor
from storesales.constants import (
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_SAMPLE_SUBMISSION_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_OIL_PATH,
    EXTERNAL_HOLIDAYS_EVENTS_PATH,
)


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


def run_study(df: pd.DataFrame, predictor: SalesPredictor, n_stores: int = 54):
    stores = np.arange(1, n_stores + 1)

    for family_group in predictor.family_groups:
        store_family_groups = list(product(stores, family_group))
        n_choices = predictor.get_n_store_family_choices(family_group)

        for store, family in random.sample(store_family_groups, n_choices):
            condition = (df["store_nbr"] == store) & (df["family"] == family)
            train = df[condition][["ds", "y", "dcoilwtico"]].copy()

            print(f"\n\nFamily: {family} - Store: {store}")

            outer_results = []
            for outer_train_index, outer_test_index in predictor.outer_cv.split(train):
                outer_train = train.iloc[outer_train_index]
                outer_test = train.iloc[outer_test_index]

                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: predictor.objective(trial, outer_train),
                    **predictor.optuna_optimize_kwargs,
                )

                rmsle_loss = predictor.evaluate_and_save_tune(
                    family_group=family_group,
                    best_params=study.best_params,
                    train=outer_train.copy(),
                    test=outer_test.copy(),
                )
                outer_results.append(rmsle_loss)
                print(
                    f"Outer: {predictor.render_model(study.best_params)} - RMSLE: {rmsle_loss}"
                )

            final_outer_rmsle = np.mean(outer_results)
            print(f"\nFamily: {family} - Store: {store} RMSLE: {final_outer_rmsle}")

        predictor.log_best(family_group)

    return predictor
