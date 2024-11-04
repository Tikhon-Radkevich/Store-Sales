from darts.timeseries import TimeSeries
from darts.models import LightGBMModel


def fit_light_gb_models(
    family_series_dict: dict[str, TimeSeries],
    future_covariates: dict[str, list[TimeSeries]],
    past_covariates: dict[str, list[TimeSeries]] | None,
) -> dict[str, LightGBMModel]:
    if past_covariates is None:
        past_covariates = {}

    light_gb_models = {}

    for family, series in family_series_dict.items():
        inputs = {
            "series": series,
            "future_covariates": future_covariates.get(family, None),
            "past_covariates": past_covariates.get(family, None),
        }
        light_gb_models[family] = LightGBMModel(
            lags=24,
            lags_future_covariates=[i for i in range(-30, 1, 3)],
            lags_past_covariates=[i for i in range(-31, -15, 3)],
        )

        light_gb_models[family].fit(**inputs)

    return light_gb_models
