from joblib import Parallel, delayed

import pandas as pd
from darts.models import RegressionModel

from storesales.loss import clipped_rmsle
from storesales.light_gbm.dataset import FamilyDataset


def parallel_prediction(
    dataset: dict[str, FamilyDataset],
    models: dict[str, RegressionModel],
    prediction_range: pd.DatetimeIndex,
    stride=1,
    parallel=False,
):
    """Ger all predicted and true values for each date in `prediction_range`"""
    def make_prediction(family: str) -> pd.DataFrame:
        series = dataset[family].series
        stores = dataset[family].stores
        family_predictions = []

        for test_date in prediction_range[::stride]:
            inputs = dataset[family].get_cut_inputs(test_date)
            preds = models[family].predict(n=16, show_warnings=False, **inputs)

            true_values = [s.slice_intersect(p) for p, s in zip(preds, series)]
            store_predictions = []

            for store, pred, true in zip(stores, preds, true_values):
                pred_df = pred.pd_series().rename("prediction")
                true_df = true.pd_series().rename("true_values")

                result_df = pd.concat([pred_df, true_df], axis=1).reset_index()
                result_df["store_nbr"] = store

                store_predictions.append(result_df)

            stores_predictions_df = pd.concat(store_predictions, axis=0)
            stores_predictions_df["date_id"] = test_date

            family_predictions.append(stores_predictions_df)

        family_predictions_df = pd.concat(family_predictions, axis=0)
        family_predictions_df["family"] = family

        return family_predictions_df

    if parallel:
        losses = Parallel(n_jobs=-1)(delayed(make_prediction)(f) for f in models.keys())
    else:
        losses = [make_prediction(f) for f in models.keys()]

    return pd.concat(losses)


def evaluate(
    dataset: dict[str, FamilyDataset],
    models: dict[str, RegressionModel],
    evaluate_range: pd.DatetimeIndex,
    stride=1,
    parallel=False,
) -> pd.DataFrame:
    def evaluate_family(family: str) -> pd.DataFrame:
        series = dataset[family].series

        multi_index = pd.MultiIndex.from_product(
            [[family], dataset[family].stores], names=["family", "store_nbr"]
        )

        family_losses = []
        for test_date in evaluate_range[::stride]:
            inputs = dataset[family].get_cut_inputs(test_date)
            preds = models[family].predict(n=16, show_warnings=False, **inputs)

            true_values = [s.slice_intersect(p) for p, s in zip(preds, series)]

            loss = [
                clipped_rmsle(t.values(), p.values())
                for t, p in zip(true_values, preds)
            ]
            series_loss = pd.Series(
                loss, index=multi_index, name=test_date.strftime("%Y.%m.%d")
            )

            family_losses.append(series_loss)

        family_losses_df = pd.concat(family_losses, axis=1)
        return family_losses_df

    if parallel:
        losses = Parallel(n_jobs=-1)(delayed(evaluate_family)(f) for f in models.keys())
    else:
        losses = [evaluate_family(f) for f in models.keys()]

    return pd.concat(losses)
