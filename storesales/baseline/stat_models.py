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
