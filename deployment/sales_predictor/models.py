import pickle

import pandas as pd
import numpy as np


def load_picle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


class TestModel:
    start_timestamp = pd.Timestamp.now().normalize()
    n = 16

    test_prediction_df = pd.DataFrame(
        {
            "date": pd.date_range(start_timestamp, periods=n, freq="D"),
            "sales": np.random.rand(n) * 100,
        }
    )

    def __init__(self, *args, **kwargs): ...

    def make_prediction(self, *_args, **_kwargs) -> pd.DataFrame:
        return self.test_prediction_df


class BaselineModel:
    def __init__(self, model_file_path: str, name: str, data_file_path: str):
        self.name = name
        self.data_file_path = data_file_path
        self.model = load_picle(model_file_path)

    def make_prediction(self, family: str, store_nbr: int) -> pd.DataFrame:
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

    def make_prediction(self, family: str, store_nbr: int) -> pd.DataFrame:
        family_dataset_file_path = self.lightgbms_info[family]["data_file_path"]

        with open(family_dataset_file_path, "rb") as file:
            family_dataset = pickle.load(file)

        inputs = family_dataset.get_inputs()

        model = self.models[family]
        predictions = model.predict(n=16, **inputs, show_warnings=False)

        store_index = family_dataset.stores.index(store_nbr)

        prediction = predictions[store_index].pd_dataframe()
        prediction["sales"] = family_dataset.scaler.inverse_transform_by_key(
            prediction["sales"], family, store_nbr
        )
        prediction.columns.name = None

        return prediction.reset_index()
