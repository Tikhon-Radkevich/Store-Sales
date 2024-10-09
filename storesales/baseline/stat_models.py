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
