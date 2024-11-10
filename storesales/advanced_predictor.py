import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from storesales.baseline.sales_predictor import SalesPredictor


class AdvancedPredictor:
    """Class to combine and compare baseline models and LightGBM."""

    def __init__(
        self,
        baseline_model_names: list[str],
        baseline_model_file_paths: list[str],
        baseline_train_df: pd.DataFrame,
        baseline_test_df: pd.DataFrame,
        lightgbm_model_loss_df: pd.DataFrame,
        lightgbm_model_prediction_df: pd.DataFrame,
        lightgbm_model_name: str = "LightGBM",
    ):
        self.baseline_model_names = baseline_model_names
        self.baseline_model_file_paths = baseline_model_file_paths
        self.baseline_train_df = baseline_train_df
        self.baseline_test_df = baseline_test_df

        self.lightgbm_model_loss_df = lightgbm_model_loss_df
        self.lightgbm_model_prediction_df = lightgbm_model_prediction_df
        self.lightgbm_model_name = lightgbm_model_name

        self._baseline_models = self._load_baseline_models()
        self._baseline_losses = self._load_baseline_losses()

        self._fit_baseline_models()
        self._baseline_predictions = self._make_baseline_predictions()

        self._combined_loss = self._get_combined_loss()
        self._combined_prediction = self._get_combined_prediction()

    def get_optimal_prediction(self, models: list[str] = None):
        combined_loss = self._combined_loss.copy()
        if models is not None:
            model_condition = combined_loss.index.get_level_values("model").isin(models)
            combined_loss = combined_loss[model_condition]

        family_store_mean_loss = combined_loss.mean(axis=1).rename("loss")
        min_loss_ids = family_store_mean_loss.groupby(["family", "store_nbr"]).idxmin()
        return self._combined_prediction.loc[min_loss_ids].copy()

    def make_family_loss_plot(self, family: str):
        mean_loss = self._combined_loss.mean(axis=1).rename("loss").reset_index()
        family_store_mean_loss = mean_loss.reset_index()
        family_store_mean_loss = family_store_mean_loss[
            family_store_mean_loss["family"] == family
        ]

        family_loss_pivot = family_store_mean_loss.pivot(
            index="store_nbr", columns="model", values="loss"
        )

        palette = sns.color_palette(
            "dark:#5A9_r", n_colors=len(family_loss_pivot.columns)
        )
        family_loss_pivot.plot(kind="bar", width=0.7, figsize=(20, 10), color=palette)
        plt.title(f"Mean Loss by Model for Each Store in Family {family}", fontsize=18)
        plt.xlabel("Store Number", fontsize=16)
        plt.ylabel("Mean Loss", fontsize=16)
        plt.legend(title="Model", fontsize=16, title_fontsize=16)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.figure()
        plt.tight_layout()
        plt.show()

    def make_overall_family_loss_plot(self):
        mean_loss = self._combined_loss.mean(axis=1).rename("loss").reset_index()
        family_mean_loss = (
            mean_loss.groupby(["family", "model"])["loss"].mean().reset_index()
        )

        family_loss_pivot = family_mean_loss.pivot(
            index="family", columns="model", values="loss"
        )

        palette = sns.color_palette(
            "dark:#5A9_r", n_colors=len(family_loss_pivot.columns)
        )

        family_loss_pivot.plot(kind="bar", width=0.7, color=palette, figsize=(20, 10))
        plt.title("Mean Loss by Model for Each Family", fontsize=18)
        plt.xlabel("Family", fontsize=16)
        plt.ylabel("Mean Loss", fontsize=16)
        plt.legend(title="Model", fontsize=16, title_fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

    def _load_baseline_losses(self):
        losses = {}
        for model_name, model in self._baseline_models.items():
            loss_file_path = model.eval_loss_csv
            loss_df = pd.read_csv(loss_file_path, index_col=["family", "store_nbr"])
            losses[model_name] = loss_df

        return losses

    def _fit_baseline_models(self):
        for model_name, model in self._baseline_models.items():
            print(f"Fitting {model_name}...")
            model.fit(self.baseline_train_df)

    def _make_baseline_predictions(self):
        predictions = {}
        for model_name, model in self._baseline_models.items():
            print(f"Predicting with {model_name}...")
            predictions[model_name] = model.predict(self.baseline_test_df)
            predictions[model_name].rename(columns={"yhat": "sales"}, inplace=True)
        return predictions

    def _get_combined_prediction(self):
        combined_prediction = {
            self.lightgbm_model_name: self.lightgbm_model_prediction_df
        }
        combined_prediction.update(self._baseline_predictions)
        combined_prediction_df = pd.concat(combined_prediction)

        combined_prediction_df.index.names = ["model", "index"]
        combined_prediction_df.reset_index(inplace=True)
        return combined_prediction_df.set_index(["model", "family", "store_nbr"])

    def _load_baseline_models(self):
        baseline_models = {}
        for model_name, file_name in zip(
            self.baseline_model_names, self.baseline_model_file_paths
        ):
            baseline_models[model_name] = SalesPredictor.load(file_name)
        return baseline_models

    def _get_combined_loss(self):
        combined_loss = {self.lightgbm_model_name: self.lightgbm_model_loss_df}
        combined_loss.update(self._baseline_losses)
        combined_loss_df = pd.concat(combined_loss)
        combined_loss_df.index.names = ["model", "family", "store_nbr"]
        return combined_loss_df
