from joblib import Parallel, delayed

import pandas as pd
from darts.models import RegressionModel

from storesales.light_gbm.loss import clipped_rmsle
from storesales.light_gbm.dataset import FamilyDataset


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
