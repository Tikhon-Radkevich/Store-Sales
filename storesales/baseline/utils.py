import warnings
import random
from collections import defaultdict
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import pandas as pd

import optuna

from storesales.light_gbm.preprocessing import preprocess
from storesales.baseline.loss import rmsle
from storesales.baseline.sales_predictor import SalesPredictor
from storesales.constants import (
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_SAMPLE_SUBMISSION_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_OIL_PATH,
    EXTERNAL_HOLIDAYS_EVENTS_PATH,
    TRAIN_TEST_SPLIT_DATE,
)


def make_time_series_dataset(
    df: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    test_size: int = 16,
):
    # train data for nested cv
    train_dataset = {
        "train": defaultdict(dict),
        "test": defaultdict(dict),
    }

    # nested cv outer loop test period
    test_period = pd.Timedelta(days=test_size)
    end_test_cutoffs = [cutoff + test_period for cutoff in cutoffs]

    family_to_store_grouped = df.groupby(["family", "store_nbr"])
    for (family, store), group in tqdm(family_to_store_grouped):
        for start_test, end_test in zip(cutoffs, end_test_cutoffs):
            if start_test <= group["ds"].min():
                continue

            train_mask = group["ds"] < start_test
            test_mask = (group["ds"] >= start_test) & (group["ds"] < end_test)

            train_data = group[train_mask]
            test_data = group[test_mask]

            train_dataset["train"][(store, family)][start_test] = train_data
            train_dataset["test"][(store, family)][start_test] = test_data

    return train_dataset


def calculate_loss_for_date(
    predictor: SalesPredictor, df: pd.DataFrame, date: pd.Timestamp
):
    train = df[df["ds"] < date]
    test = df[(df["ds"] >= date) & (df["ds"] < date + pd.Timedelta(days=16))]

    predictor.fit(train, disable_tqdm=True)
    prediction = predictor.predict(test, disable_tqdm=True)

    loss = rmsle(prediction["y"], prediction["yhat"])
    return loss


def evaluate(df: pd.DataFrame, predictor: SalesPredictor, n_jobs: int = -1):
    series_test_range = pd.date_range(TRAIN_TEST_SPLIT_DATE, df["ds"].max(), freq="D")
    losses = Parallel(n_jobs=n_jobs)(
        delayed(calculate_loss_for_date)(predictor, df, date)
        for date in tqdm(series_test_range)
    )
    return np.mean(losses)


def run_study(dataset, predictor: SalesPredictor, optuna_log_off=True):
    if optuna_log_off:
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    for family_group in predictor.family_groups:
        print(f"Family Group: {family_group}:")

        n_choices = predictor.get_n_store_family_choices(family_group)

        store_family_pairs = predictor.store_family_pairs[family_group]

        if len(store_family_pairs) <= n_choices:
            sampled_store_family = store_family_pairs
        else:
            sampled_store_family = random.sample(store_family_pairs, n_choices)

        family_group_loss = []
        for i_sample, (store, family) in tqdm(
            enumerate(sampled_store_family), total=len(sampled_store_family)
        ):
            for start_test, outer_train in dataset["train"][(store, family)].items():
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: predictor.objective(trial, outer_train),
                    **predictor.optuna_optimize_kwargs,
                )

                test_loss = []
                for _store, _family in store_family_pairs:
                    _outer_test = dataset["test"][(_store, _family)].get(
                        start_test, None
                    )
                    _outer_train = dataset["train"][(_store, _family)].get(
                        start_test, None
                    )

                    if _outer_test is None or _outer_train is None:
                        continue

                    model = predictor.get_best_model(study.best_params)
                    model.fit(_outer_train)
                    forecast = model.predict(_outer_test)
                    y_pred = forecast["yhat"].values
                    y_true = _outer_test["y"].values
                    loss = rmsle(y_true, y_pred)
                    test_loss.append(loss)
                    predictor.store_family_loss_storage[(_store, _family)].append(loss)

                predictor.update_tune_loss_storage(
                    family_group, test_loss, i_sample, start_test
                )

                fold_test_loss = np.mean(test_loss)
                family_group_loss.append(fold_test_loss)

                predictor.log_study(
                    family_group=family_group,
                    best_params=study.best_params,
                    loss=fold_test_loss,
                )
        predictor.calc_and_log_mean_params(family_group)
        print(f"RMSLE: {np.mean(family_group_loss)}\n")

    losses = [
        value["loss"]
        for _key, value in predictor.family_to_model_params_storage.items()
    ]
    print(f"\n\nTotal RMSLE: {np.mean(losses)} \n")
    return predictor


def load_baseline_data(use_light_gbm_preprocessing=False):
    # load oil
    oil_df = pd.read_csv(EXTERNAL_OIL_PATH, parse_dates=["date"])
    oil_df.set_index("date", inplace=True)
    oil_df = oil_df.asfreq("D")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill()
    oil_df = oil_df.dropna()

    # load train
    original_train_df = pd.read_csv(EXTERNAL_TRAIN_PATH, parse_dates=["date"])
    original_train_df.sort_values(by=["date", "store_nbr", "family"], inplace=True)

    if use_light_gbm_preprocessing:
        original_train_df = preprocess(original_train_df)

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
