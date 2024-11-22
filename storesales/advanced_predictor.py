import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from storesales.baseline.sales_predictor import SalesPredictor


class AdvancedPredictor:
    """Class to combine and compare baseline models and LightGBM."""

    def __init__(
        self,
        loss_split_date_str: str,
        baseline_model_names: list[str],
        baseline_model_file_paths: list[str],
        baseline_train_df: pd.DataFrame,
        baseline_test_df: pd.DataFrame,
        lightgbm_model_loss_df: pd.DataFrame,
        lightgbm_model_prediction_df: pd.DataFrame,
        lightgbm_model_name: str = "LightGBM",
    ):
        self.loss_split_date_str = loss_split_date_str
        self.baseline_model_names = baseline_model_names
        self.baseline_model_file_paths = baseline_model_file_paths
        self.baseline_train_df = baseline_train_df
        self.baseline_test_df = baseline_test_df

        self.lightgbm_model_loss_df = lightgbm_model_loss_df
        self.lightgbm_model_prediction_df = lightgbm_model_prediction_df
        self.lightgbm_model_name = lightgbm_model_name

        self.min_loss_ids = None

        self._baseline_models = self._load_baseline_models()
        self._baseline_losses = self._load_baseline_losses()

        self._fit_baseline_models()
        self._baseline_predictions = self._make_baseline_predictions()

        self._combined_loss = self._get_combined_loss()
        self._combined_prediction = self._get_combined_prediction()

        self._loss_to_choose_model_df, self._test_loss_df = self._split_loss()

    def get_min_loss(self, test_loss: bool = True):
        loss_df = self._test_loss_df if test_loss else self._loss_to_choose_model_df
        family_store_mean_loss = loss_df.mean(axis=1).rename("loss")
        min_loss_mean = (
            family_store_mean_loss.groupby(["family", "store_nbr"]).min().mean()
        )
        return min_loss_mean

    def get_mean_test_loss(self):
        return self._test_loss_df.mean(axis=1).rename("loss").reset_index()

    def get_mean_model_choose_loss(self):
        return self._loss_to_choose_model_df.mean(axis=1).rename("loss").reset_index()

    def get_std_test_loss(self):
        return self._test_loss_df.std(axis=1).rename("std").reset_index()

    def get_std_model_choose_loss(self):
        return self._loss_to_choose_model_df.std(axis=1).rename("std").reset_index()

    def filter_combined_loss(
        self,
        loss_df: pd.DataFrame,
        models: list[str] = None,
        lightgbm_drop_families: list[str] = None,
    ):
        if models is not None:
            model_condition = loss_df.index.get_level_values("model").isin(models)
            loss_df = loss_df[model_condition]

        if lightgbm_drop_families is not None:
            family_levels = loss_df.index.get_level_values("family")
            model_levels = loss_df.index.get_level_values("model")

            family_condition = family_levels.isin(lightgbm_drop_families)
            model_condition = model_levels == self.lightgbm_model_name
            mask = family_condition & model_condition

            loss_df = loss_df[~mask]

        return loss_df

    def get_optimal_model_ids(
        self, models: list[str] = None, lightgbm_drop_families: list[str] = None
    ):
        combined_loss = self.filter_combined_loss(
            loss_df=self._loss_to_choose_model_df.copy(),
            models=models,
            lightgbm_drop_families=lightgbm_drop_families,
        )

        # Calculate mean loss across time
        family_store_mean_loss = combined_loss.mean(axis=1).rename("loss")

        first_strategy_min_ids = family_store_mean_loss.groupby(
            ["family", "store_nbr"]
        ).idxmin()

        grouped_mean_loss = family_store_mean_loss.groupby(
            ["model", "family"]
        ).transform("mean")
        second_strategy_min_ids = grouped_mean_loss.groupby(
            ["family", "store_nbr"]
        ).idxmin()

        first_loss = (
            self._test_loss_df.loc[first_strategy_min_ids]
            .mean(axis=1)
            .groupby("family")
            .mean()
        )
        second_loss = (
            self._test_loss_df.loc[second_strategy_min_ids]
            .mean(axis=1)
            .groupby("family")
            .mean()
        )
        better_strategy = pd.Series(first_loss < second_loss, name="use_first")

        optimal_indices = np.concatenate(
            [
                first_strategy_min_ids[family].values
                if better_strategy.loc[family]
                else second_strategy_min_ids[family].values
                for family in better_strategy.index
            ]
        )
        return pd.MultiIndex.from_tuples(
            optimal_indices, names=["model", "family", "store_nbr"]
        )

    def get_optimal_prediction(
        self,
        models: list[str] = None,
        lightgbm_drop_families: list[str] = None,
    ):
        """
        Get optimal predictions based on minimum loss.

        Args:
            models (list[str]): List of models to consider for selection. Default is None (consider all models).
            lightgbm_drop_families (list[str]): List of families to exclude from LightGBM selection. Default is None.
        """

        self.min_loss_ids = self.get_optimal_model_ids(models, lightgbm_drop_families)

        return self._combined_prediction.loc[self.min_loss_ids].copy()

    def make_model_selection_plot(self):
        """
        Create a bar plot showing the number of times each model is selected
        for each family based on the minimum loss.
        """
        # Extract family and model from self.min_loss_ids
        selection_counts = (
            self.min_loss_ids.to_frame(index=False)
            .groupby(["family", "model"])
            .size()
            .reset_index(name="count")
        )

        selection_pivot = selection_counts.pivot(
            index="family", columns="model", values="count"
        ).fillna(0)

        palette = sns.color_palette(
            "dark:#5A9_r", n_colors=len(selection_pivot.columns)
        )
        selection_pivot.plot(kind="bar", width=0.7, color=palette, figsize=(20, 10))

        plt.title("Number of Models Selected per Family", fontsize=18)
        plt.xlabel("Family", fontsize=16)
        plt.ylabel("Selection Count", fontsize=16)
        plt.legend(title="Model", fontsize=16, title_fontsize=16)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

    def make_family_loss_plot(self, family: str, test_loss: bool = True):
        if test_loss:
            mean_loss = self.get_mean_test_loss()
        else:
            mean_loss = self.get_mean_model_choose_loss()

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

    def make_overall_family_loss_plot(self, test_loss: bool = True):
        if test_loss:
            mean_loss = self.get_mean_test_loss()
            std_loss = self.get_std_test_loss()
        else:
            mean_loss = self.get_mean_model_choose_loss()
            std_loss = self.get_std_model_choose_loss()

        # Calculate mean and variance grouped by family and model
        family_mean_loss = mean_loss.groupby(["family", "model"])["loss"].mean().reset_index()
        family_variance_loss = std_loss.groupby(["family", "model"])["std"].mean().reset_index()

        # Merge mean and variance data
        family_loss_data = family_mean_loss.merge(
            family_variance_loss, on=["family", "model"]
        )

        # Pivot for plotting
        mean_pivot = family_loss_data.pivot(index="family", columns="model", values="loss")
        std_pivot = family_loss_data.pivot(index="family", columns="model", values="std")
        std_pivot = mean_pivot / std_pivot

        # Create the first plot for mean loss
        colors = sns.color_palette("dark:#5A9_r", n_colors=len(mean_pivot.columns))
        mean_pivot.plot(kind="bar", width=0.7, color=colors, figsize=(20, 10))
        plt.title("Mean Loss by Model for Each Family", fontsize=18)
        plt.xlabel("Family", fontsize=16)
        plt.ylabel("Mean Loss", fontsize=16)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

        # Create the second plot for variance
        std_pivot.plot(kind="bar", width=0.7, color=colors, figsize=(20, 10))
        plt.title("Std by Model for Each Family", fontsize=18)
        plt.xlabel("Family", fontsize=16)
        plt.ylabel("Std", fontsize=16)
        plt.xticks(rotation=90, fontsize=16)
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
        lightgbm_pred_df = self.lightgbm_model_prediction_df.rename(
            columns={"date": "ds"}
        )
        combined_prediction = {self.lightgbm_model_name: lightgbm_pred_df}
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

    def _split_loss(self):
        split_index = self._combined_loss.columns.get_loc(self.loss_split_date_str)
        _loss_to_choose_model_df = self._combined_loss.iloc[:, :split_index]
        _test_loss_df = self._combined_loss.iloc[:, split_index:]
        return _loss_to_choose_model_df, _test_loss_df
