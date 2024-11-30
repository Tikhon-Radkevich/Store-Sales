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

    def calc_mean_loss(self, indices: pd.MultiIndex, test_loss: bool = True):
        loss_df = self._test_loss_df if test_loss else self._loss_to_choose_model_df
        return loss_df.loc[indices].mean(axis=1).groupby("family").mean()

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
        self,
        models: list[str] = None,
        lightgbm_drop_families: list[str] = None,
        strategy: str = "each_store",
        use_std: bool = False,  # todo: remove
    ) -> pd.MultiIndex:
        strategies = ("each_store", "mean_family", "combined")
        if strategy not in strategies:
            raise ValueError(f"Invalid strategy. Choose from {strategies}")

        def get_multiindex(indices: list) -> pd.MultiIndex:
            return pd.MultiIndex.from_tuples(
                indices, names=["model", "family", "store_nbr"]
            )

        def calc_mean_test_loss(indices: pd.MultiIndex) -> pd.Series:
            return self._test_loss_df.loc[indices].mean(axis=1).groupby("family").mean()

        def get_each_store_strategy_min_ids(loss_df: pd.Series) -> pd.MultiIndex:
            idx_min_series = loss_df.groupby(["family", "store_nbr"]).idxmin()
            return get_multiindex(idx_min_series.values)

        def get_mean_family_strategy_min_ids(loss_df: pd.Series) -> pd.MultiIndex:
            grouped_loss = loss_df.groupby(["model", "family"])
            transformed_mean_loss = grouped_loss.transform("mean")
            idx_min_series = transformed_mean_loss.groupby(
                ["family", "store_nbr"]
            ).idxmin()
            return get_multiindex(idx_min_series.values)

        def combine_strategies(strategies_ids: list[pd.MultiIndex]) -> pd.MultiIndex:
            strategies_losses = [calc_mean_test_loss(ids) for ids in strategies_ids]

            stacked_losses = pd.concat(strategies_losses, axis=1).stack().rename("loss")
            stacked_losses = stacked_losses.rename_axis(["family", "i_strategy"])

            min_loss_indices = stacked_losses.groupby("family").idxmin()

            optimal_indices = [
                strategies_ids[i_strategy][
                    strategies_ids[i_strategy].get_level_values("family") == family
                ]
                for family, i_strategy in min_loss_indices.values
            ]

            return get_multiindex(np.concatenate(optimal_indices))

        combined_loss = self.filter_combined_loss(
            loss_df=self._loss_to_choose_model_df.copy(),
            models=models,
            lightgbm_drop_families=lightgbm_drop_families,
        )

        # Calculate mean loss across time
        family_store_mean_loss = combined_loss.mean(axis=1).rename("loss")

        if use_std:
            family_store_mean_loss *= combined_loss.std(axis=1)

        if strategy == "each_store":
            return get_each_store_strategy_min_ids(family_store_mean_loss)

        elif strategy == "mean_family":
            return get_mean_family_strategy_min_ids(family_store_mean_loss)

        elif strategy == "combined":
            each_store_strategy_min_ids = get_each_store_strategy_min_ids(
                family_store_mean_loss
            )
            mean_family_strategy_min_ids = get_mean_family_strategy_min_ids(
                family_store_mean_loss
            )
            strategies_min_ids = [
                each_store_strategy_min_ids,
                mean_family_strategy_min_ids,
            ]
            return combine_strategies(strategies_min_ids)

        else:
            raise ValueError(f"Invalid strategy. Choose from {strategies}")

    def get_loss_comparison(self, test_loss: bool = True, use_std: bool = False):
        each_store_strategy_ids = self.get_optimal_model_ids(
            strategy="each_store", use_std=use_std
        )
        mean_family_strategy_ids = self.get_optimal_model_ids(
            strategy="mean_family", use_std=use_std
        )

        loss_df = self._test_loss_df if test_loss else self._loss_to_choose_model_df

        each_store_strategy_loss = (
            loss_df.loc[each_store_strategy_ids]
            .mean(axis=1)
            .groupby("family")
            .mean()
            .rename("each_store")
        )
        mean_family_strategy_loss = (
            loss_df.loc[mean_family_strategy_ids]
            .mean(axis=1)
            .groupby("family")
            .mean()
            .rename("mean_family")
        )
        total_df = pd.concat(
            [each_store_strategy_loss, mean_family_strategy_loss], axis=1
        )
        return total_df.stack().rename_axis(["family", "strategy"]).rename("loss")

    def get_optimal_prediction(
        self,
        models: list[str] = None,
        lightgbm_drop_families: list[str] = None,
        strategy: str = "each_store",
        use_std: bool = False,
    ):
        """
        Get optimal predictions based on minimum loss.
        """

        min_loss_ids = self.get_optimal_model_ids(
            models, lightgbm_drop_families, strategy, use_std
        )

        return self._combined_prediction.loc[min_loss_ids].copy()

    @staticmethod
    def make_model_selection_plot(models_ids, families=None):
        """
        Create a bar plot showing the number of times each model is selected
        for each family based on the minimum loss.
        """
        # Extract family and model from self.min_loss_ids
        selection_counts = (
            models_ids.to_frame(index=False)
            .groupby(["family", "model"])
            .size()
            .reset_index(name="count")
        )

        selection_pivot = selection_counts.pivot(
            index="family", columns="model", values="count"
        ).fillna(0)

        if families is not None:
            selection_pivot = selection_pivot.loc[families]

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
        loss_df = self._test_loss_df if test_loss else self._loss_to_choose_model_df
        mean_loss = loss_df.mean(axis=1).rename("loss").reset_index()

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

    def make_overall_family_loss_plot(
        self, families: list[str] = None, test_loss: bool = True, plot_std: bool = False
    ):
        loss_df = self._test_loss_df if test_loss else self._loss_to_choose_model_df
        if families is not None:
            loss_df = loss_df[loss_df.index.get_level_values("family").isin(families)]

        mean_loss = loss_df.mean(axis=1).rename("loss").reset_index()
        std_loss = loss_df.std(axis=1).rename("std").reset_index()

        # Calculate mean and variance grouped by family and model
        family_mean_loss = (
            mean_loss.groupby(["family", "model"])["loss"].mean().reset_index()
        )
        family_std_loss = (
            std_loss.groupby(["family", "model"])["std"].mean().reset_index()
        )

        # Merge mean and variance data
        family_loss_data: pd.DataFrame = family_mean_loss.merge(
            family_std_loss, on=["family", "model"]
        )

        # Pivot for plotting
        mean_pivot = family_loss_data.pivot(
            index="family", columns="model", values="loss"
        )
        std_pivot = family_loss_data.pivot(
            index="family", columns="model", values="std"
        )
        # std_pivot *= mean_pivot

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

        # Create the second plot for std
        if plot_std:
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
