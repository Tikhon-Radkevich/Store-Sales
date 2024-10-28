import numpy as np
import pandas as pd


class DailyMeanModel:
    def __init__(self, window: int):
        self.window = window
        self.mean_prediction = None

    def fit(self, train: pd.DataFrame) -> None:
        self.mean_prediction = train["y"].tail(self.window).mean()
        if pd.isna(self.mean_prediction):
            self.mean_prediction = 0

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.mean_prediction is None:
            raise ValueError("Model not fitted yet.")
        future_predictions = future.copy()
        future_predictions["yhat"] = self.mean_prediction
        return future_predictions


class DayOfWeekMeanModel:
    def __init__(self, weekdays_window: int, weekends_window: int):
        self.weekdays_window = weekdays_window
        self.weekends_window = weekends_window
        self.mean_predictions = None

    def fit(self, train: pd.DataFrame) -> None:
        weekdays_data = train[train["ds"].dt.weekday < 5]
        weekends_data = train[train["ds"].dt.weekday >= 5]
        self.mean_predictions = {
            "weekdays": weekdays_data["y"].tail(self.weekdays_window).mean(),
            "weekends": weekends_data["y"].tail(self.weekends_window).mean(),
        }

        if pd.isna(self.mean_predictions["weekdays"]):
            self.mean_predictions["weekdays"] = 0
        if pd.isna(self.mean_predictions["weekends"]):
            self.mean_predictions["weekends"] = 0

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.mean_predictions is None:
            raise ValueError("Model not fitted yet.")
        future_predictions = future.copy()
        future_predictions["yhat"] = [
            self.mean_predictions["weekdays"]
            if day < 5
            else self.mean_predictions["weekends"]
            for day in future["ds"].dt.weekday
        ]
        return future_predictions


class WeightedDayMeanModel:
    def __init__(
        self,
        days_window: int,
        weeks_window: int,
        months_window: int,
        years_window: int,
        day_weight: float,
        week_weight: float,
        month_weight: float,
        year_weight: float,
        bias: float = 0,
    ):
        self.days_window = days_window
        self.weeks_window = weeks_window
        self.months_window = months_window
        self.years_window = years_window

        self.day_weight = day_weight
        self.week_weight = week_weight
        self.month_weight = month_weight
        self.year_weight = year_weight

        self.bias = bias

        self.daily_mean_predictions = None
        self.train = None

        self._process_weights()

    def fit(self, train: pd.DataFrame) -> None:
        self.train = train.set_index("ds")

        self.daily_mean_predictions = train["y"].tail(self.days_window).mean()

        if pd.isna(self.daily_mean_predictions):
            self.days_window = 0
            self._process_weights()
            self.daily_mean_predictions = 0

    def _process_weights(self):
        if self.days_window == 0:
            self.day_weight = 0
        if self.weeks_window == 0:
            self.week_weight = 0
        if self.months_window == 0:
            self.month_weight = 0
        if self.years_window == 0:
            self.year_weight = 0

        total_weight = (
            self.day_weight + self.week_weight + self.month_weight + self.year_weight
        )
        self.day_weight /= total_weight
        self.week_weight /= total_weight
        self.month_weight /= total_weight
        self.year_weight /= total_weight

    def _get_past_averages(self, dates: list[pd.Timestamp]) -> float:
        # Averages for weeks, months, years ago
        past_data = self.train.loc[self.train.index.isin(dates), "y"]
        return past_data.mean() if not past_data.empty else 0.0

    def _get_day_averages(self, date: pd.Timestamp) -> tuple[float, float, float]:
        # Precompute past dates based on windows
        week_ago_dates = [
            date - pd.DateOffset(weeks=i) for i in range(1, self.weeks_window + 1)
        ]
        month_ago_dates = [
            date - pd.DateOffset(months=i) for i in range(1, self.months_window + 1)
        ]
        year_ago_dates = [
            date - pd.DateOffset(years=i) for i in range(1, self.years_window + 1)
        ]

        week_avg = self._get_past_averages(week_ago_dates)
        month_avg = self._get_past_averages(month_ago_dates)
        year_avg = self._get_past_averages(year_ago_dates)

        return week_avg, month_avg, year_avg

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.train is None:
            raise ValueError("Model not fitted yet.")

        future = future.copy()

        yhat_values = []
        for future_date in future["ds"]:
            week_avg, month_avg, year_avg = self._get_day_averages(future_date)

            yhat = (
                (self.daily_mean_predictions * self.day_weight)
                + (week_avg * self.week_weight)
                + (month_avg * self.month_weight)
                + (year_avg * self.year_weight)
                + self.bias
            )
            yhat = np.clip(yhat, 0, None)
            yhat_values.append(yhat)

        future["yhat"] = yhat_values
        return future
