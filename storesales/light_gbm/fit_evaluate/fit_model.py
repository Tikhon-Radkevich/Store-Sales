# todo - remove this file

# from darts.models import LightGBMModel

# from storesales.light_gbm.dataset import FamilyDataset
# from storesales.light_gbm


# def fit_light_gb_model(dataset: FamilyDataset, **model_kwargs):
#     light_gb_model = LightGBMModel(
#         lags=24,
#         lags_future_covariates=[i for i in range(-9, 1, 3)],
#         lags_past_covariates=[i for i in range(-25, -15, 3)],
#         **model_kwargs,
#     )
#
#     inputs = {
#         "series": dataset.series,
#         "future_covariates": dataset.future_covariates,
#         "past_covariates": dataset.past_covariates,
#     }
#
#     light_gb_model.fit(**inputs)
#
#     return light_gb_model


# def fit_light_gb_models(
#     dataset: dict[str, FamilyDataset],
#     **model_kwargs,
# ) -> dict[str, LightGBMModel]:
#     light_gb_models = {}
#
#     for family, family_dataset in dataset.items():
#         light_gb_models[family] = LightGBMModel(
#             lags=24,
#             lags_future_covariates=[i for i in range(-9, 1, 3)],
#             lags_past_covariates=[i for i in range(-25, -15, 3)],
#             **model_kwargs,
#         )
#         inputs = family_dataset.get_train_inputs()
#         light_gb_models[family].fit(**inputs)
#
#     return light_gb_models
