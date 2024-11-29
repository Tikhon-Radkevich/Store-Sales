import pandas as pd


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
