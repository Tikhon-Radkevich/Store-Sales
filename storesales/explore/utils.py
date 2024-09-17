import pandas as pd
import matplotlib.pyplot as plt


def plot_corr_per_store(df: pd.DataFrame, title: str, reverse_yaxis: bool = False):
    plt.figure(figsize=(20, 6))
    df.plot(kind="bar", rot=90)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Store Number", fontsize=16)
    plt.ylabel("Correlation between Sales and Transactions", fontsize=16)
    plt.title(title, fontsize=24)

    if reverse_yaxis:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def compute_weekly_corr(group: pd.Series, columns: list[str]) -> float:
    return group[columns].corr().iloc[0, 1]
