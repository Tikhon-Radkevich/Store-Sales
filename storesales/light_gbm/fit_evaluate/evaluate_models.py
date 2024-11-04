from joblib import Parallel, delayed

import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel

from storesales.light_gbm.loss import clipped_rmsle
from storesales.constants import END_TRAIN_DATE, TRAIN_TEST_SPLIT_DATE


eval_end_date = pd.Timestamp(END_TRAIN_DATE) - pd.Timedelta(days=16)
series_test_range = pd.date_range(TRAIN_TEST_SPLIT_DATE, eval_end_date, freq="D")


def evaluate(
        family_to_series_dict: dict[str, list[TimeSeries]],
        family_to_series_id_dict: dict[str, list[dict[str, int]]],
        models: dict[str, RegressionModel],
        future_covariates,
        past_covariates
) -> pd.DataFrame:
    def evaluate_family(family: str) -> pd.DataFrame:
        series = family_to_series_dict[family]
        lgb_family_stores = [
            element["store_nbr"] for element in family_to_series_id_dict[family]
        ]
        multi_index = pd.MultiIndex.from_product(
            [[family], lgb_family_stores], names=["family", "store_nbr"]
        )

        family_losses = []
        for test_date in series_test_range:
            inputs = {
                "series": [s.drop_after(test_date) for s in series],
                "future_covariates": future_covariates[family],
                "past_covariates": past_covariates[family],
            }

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

    losses = Parallel(n_jobs=-1)(
        delayed(evaluate_family)(family) for family in family_to_series_dict.keys()
    )

    return pd.concat(losses)
