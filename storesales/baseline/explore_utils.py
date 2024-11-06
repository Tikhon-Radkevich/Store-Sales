import pandas as pd
import matplotlib.pyplot as plt

from storesales.baseline.sales_predictor import SalesPredictor


def family_store_con(df, f: str, s_n: int):
    return df[(df["family"] == f) & (df["store_nbr"] == s_n)].copy()


def store_family_prediction_plot(
    predictor: SalesPredictor,
    data_to_plot: pd.DataFrame,
    family: str,
    store_nbr: int,
    test_data=None,
    data_slice: int = 1000,
):
    if test_data is None:
        train_data = data_to_plot[:-16]
        test_data = data_to_plot[-16:]
    else:
        train_data = data_to_plot

    predictor.fit(train_data)
    pred_df = predictor.predict(test_data)

    data_to_plot = data_to_plot[-data_slice:]

    plt.figure(figsize=(20, 10))
    plt.plot(data_to_plot["ds"], data_to_plot["y"], label="Actual")

    plt.plot(pred_df["ds"], pred_df["yhat"], label="Predicted")

    plt.title(f"Store {store_nbr}, Family {family}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
