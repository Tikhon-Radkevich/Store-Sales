import pickle

import pandas as pd
import numpy as np


def load_picle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


test_dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=16, freq="D")
prediction_values = np.random.rand(16) * 100

test_prediction_df = pd.DataFrame({
    "date": test_dates,
    "sales": prediction_values
})


class TestBaselineModel:
    def __init__(self, model_file_path: str, name: str, data_file_path: str):
        ...

    def make_prediction(self, family: str, store_nbr: int):
        return test_prediction_df


class TestLightGBM:
    def __init__(self, lightgbms_info: dict):
        ...

    def make_prediction(self, family: str, store_nbr: int):
        return test_prediction_df


class BaselineModel:
    def __init__(self, model_file_path: str, name: str, data_file_path: str):
        self.name = name
        self.data_file_path = data_file_path
        self.model = load_picle(model_file_path)

    def make_prediction(self, family: str, store_nbr: int):
        data_df = pd.read_csv(self.data_file_path, parse_dates=["ds"])
        mask = (data_df["family"] == family) & (data_df["store_nbr"] == store_nbr)
        family_data_df = data_df[mask]

        self.model.fit(family_data_df, disable_tqdm=True)

        forecast_start = family_data_df["ds"].max() + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.Timedelta(days=15)
        forecast = pd.date_range(forecast_start, forecast_end, freq="D")
        forecast = forecast.to_frame(name="date")
        forecast[["family", "store_nbr"]] = family, store_nbr

        prediction = self.model.predict(forecast, disable_tqdm=True)
        prediction.rename(columns={"yhat": "sales"}, inplace=True)

        return prediction[["date", "sales"]]


class LightGBM:
    def __init__(self, lightgbms_info: dict):
        self.lightgbms_info = lightgbms_info

        self.models = self._load_models()

    def _load_models(self):
        models = {}

        for family, info in self.lightgbms_info.items():
            model_file_path = info["model_file_path"]
            models[family] = load_picle(model_file_path)
        return models

    def make_prediction(self, family: str, store_nbr: int):
        family_dataset_file_path = self.lightgbms_info[family]["data_file_path"]

        with open(family_dataset_file_path, "rb") as file:
            family_dataset = pickle.load(file)

        # todo: process series before to use family_dataset.get_inputs();
        # todo: add scaling.
        cut_timestamp = pd.Timestamp("2017-04-12")
        inputs = family_dataset.get_cut_inputs(cut_timestamp)

        model = self.models[family]
        predictions = model.predict(n=16, **inputs, show_warnings=False)

        store_index = family_dataset.stores.index(store_nbr)

        prediction = predictions[store_index].pd_dataframe()
        prediction.columns.name = None

        return prediction.reset_index()


class SalesPredictor:
    def __init__(
        self,
        baseline_predictor,
        lightgbm_predictor,
        lightgbms_info: dict,
        baseline_model_paths: list[str],
        baseline_model_names: list[str],
        baseline_data_file_path: str,
        family_store_to_model_csv_file_path: str,
    ):
        self.baseline_predictor = baseline_predictor
        self.lightgbm_predictor = lightgbm_predictor

        self.lightgbms_info = lightgbms_info
        self.baseline_model_paths = baseline_model_paths
        self.baseline_model_names = baseline_model_names
        self.baseline_data_file_path = baseline_data_file_path
        self.family_store_to_model_csv_file_path = family_store_to_model_csv_file_path

        self.family_store_to_model_dict = self.load_family_store_to_model_dict()
        self.models = self.load_models()

    def make_prediction(self, family: str, store_nbr: int):
        model_name = self.family_store_to_model_dict[(family, store_nbr)]
        prediction = self.models[model_name].make_prediction(family, store_nbr)
        return {"model": model_name, "prediction": prediction.to_dict()}

    def load_models(self):
        models = {}

        for file_path, model_name in zip(
            self.baseline_model_paths, self.baseline_model_names
        ):
            models[model_name] = self.baseline_predictor(
                file_path, model_name, self.baseline_data_file_path
            )

        models["LightGBM"] = self.lightgbm_predictor(self.lightgbms_info)

        return models

    def load_family_store_to_model_dict(self):
        file_path = self.family_store_to_model_csv_file_path
        index_cols = ["family", "store_nbr"]
        family_store_to_model = pd.read_csv(file_path, index_col=index_cols)
        return family_store_to_model.to_dict()["model"]
