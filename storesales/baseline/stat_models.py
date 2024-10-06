import pandas as pd


class DailyMeanModel:
    def __init__(self, window: int):
        self.window = window
        self.prediction = None

    def fit(self, train: pd.DataFrame) -> None:
        self.prediction = train["y"].tail(self.window).mean()

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.prediction is None:
            raise ValueError("Model not fitted")
        prediction = future.copy()
        prediction["yhat"] = [self.prediction] * len(future)
        return prediction


class DayOfWeekMeanModel:
    def __init__(self, weekdays_window: int, weekends_window: int):
        self.weekdays_window = weekdays_window
        self.weekends_window = weekends_window
        self.prediction = None

    def fit(self, train: pd.DataFrame) -> None:
        weekdays = train[train["ds"].dt.weekday < 5]
        weekends = train[train["ds"].dt.weekday >= 5]
        self.prediction = {
            "weekdays": weekdays["y"].tail(self.weekdays_window).mean(),
            "weekends": weekends["y"].tail(self.weekends_window).mean(),
        }

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.prediction is None:
            raise ValueError("Model not fitted")
        prediction = future.copy()
        prediction["yhat"] = [
            self.prediction["weekdays"] if day < 5 else self.prediction["weekends"]
            for day in future["ds"].dt.weekday
        ]
        return prediction
