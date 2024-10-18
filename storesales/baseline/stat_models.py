import pandas as pd


class DailyMeanModel:
    def __init__(self, window: int):
        self.window = window
        self.mean_prediction = None

    def fit(self, train: pd.DataFrame) -> None:
        self.mean_prediction = train["y"].tail(self.window).mean()

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.mean_prediction is None:
            raise ValueError("Model not fitted yet.")
        future_predictions = future.copy()
        future_predictions["yhat"] = [self.mean_prediction] * len(future)
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
        weeks_window: int,
        months_window: int,
        years_window: int,
        week_weight: float,
        month_weight: float,
        year_weight: float,
    ):
        self.weeks_window = weeks_window
        self.months_window = months_window
        self.years_window = years_window
        self.week_weight = week_weight
        self.month_weight = month_weight
        self.year_weight = year_weight
        self.train = None

    def fit(self, train: pd.DataFrame) -> None:
        self.train = train.copy()

    def _get_day_averages(self, date: pd.Timestamp):
        # Get past data for the same day of the week, month, and year ago
        week_ago_dates = [
            date - pd.DateOffset(weeks=i) for i in range(1, self.weeks_window + 1)
        ]
        month_ago_dates = [
            date - pd.DateOffset(months=i) for i in range(1, self.months_window + 1)
        ]
        year_ago_dates = [
            date - pd.DateOffset(years=i) for i in range(1, self.years_window + 1)
        ]

        week_ago_data = self.train[self.train["ds"].isin(week_ago_dates)]["y"].mean()
        month_ago_data = self.train[self.train["ds"].isin(month_ago_dates)]["y"].mean()
        year_ago_data = self.train[self.train["ds"].isin(year_ago_dates)]["y"].mean()

        return week_ago_data, month_ago_data, year_ago_data

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.train is None:
            raise ValueError("Model not fitted yet.")

        future_predictions = future.copy()

        yhat_values = []
        for future_date in future_predictions["ds"]:
            week_avg, month_avg, year_avg = self._get_day_averages(future_date)

            yhat = (
                (week_avg * self.week_weight)
                + (month_avg * self.month_weight)
                + (year_avg * self.year_weight)
            )
            yhat_values.append(yhat)

            # update train to use prediction for future predictions
            self.train = pd.concat(
                [self.train, pd.DataFrame({"ds": [future_date], "y": [yhat]})],
                ignore_index=True,
            )

        future_predictions["yhat"] = yhat_values
        return future_predictions
