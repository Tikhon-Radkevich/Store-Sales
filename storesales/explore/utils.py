import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from storesales.constants import START_SUBMISSION_DATE, END_SUBMISSION_DATE


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


def get_holidays_on_sales_fig(
    sales_df: pd.DataFrame, holidays_df: pd.DataFrame
) -> go.Figure:
    locale_color_map = {
        "National": "#1f77b4",  # Light blue
        "Local": "#2ca02c",  # Bright green
        "Regional": "#ff7f0e",  # Orange
    }

    type_color_map = {
        "Holiday": "#9467bd",  # Purple
        "Event": "#d62728",  # Red
        "Transfer": "#bcbd22",  # Yellow-green
        "Additional": "#17becf",  # Cyan
        "Bridge": "#8c564b",  # Brown
        "Work Day": "#e377c2",  # Pink
    }

    fig = go.Figure()

    # Add sales line trace
    fig.add_trace(
        go.Scatter(
            x=sales_df["date"],
            y=sales_df["sales"],
            mode="lines",
            name="Sales",
            line=dict(color="blue"),
            legendgroup="Sales",
        )
    )

    # Add holiday dots grouped by locale
    for locale, color in locale_color_map.items():
        filtered_df = holidays_df[holidays_df["locale"] == locale]
        locale_y = [max(sales_df["sales"])] * len(filtered_df)

        fig.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=locale_y,
                mode="markers",
                name=locale,
                marker=dict(color=color, size=10, symbol="circle"),
                text=filtered_df["description"],
                customdata=np.stack([filtered_df["locale"]], axis=-1),
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br><extra></extra>",
                showlegend=True,
                legendgroup=locale,
            )
        )

    # Add holiday dots grouped by type
    for holiday_type, color in type_color_map.items():
        filtered_df = holidays_df[holidays_df["type"] == holiday_type]

        fig.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=[min(sales_df["sales"])] * len(filtered_df),
                mode="markers",
                name=holiday_type,
                marker=dict(color=color, size=10, symbol="diamond"),
                text=filtered_df["type"],
                customdata=np.stack([filtered_df["type"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
                showlegend=True,
                legendgroup=holiday_type,
            )
        )

    # Add vertical lines for test period
    fig.add_vline(
        x=START_SUBMISSION_DATE, line_width=2, line_dash="dash", line_color="white"
    )
    fig.add_vline(
        x=END_SUBMISSION_DATE, line_width=2, line_dash="dash", line_color="white"
    )

    fig.add_annotation(
        x=pd.to_datetime(START_SUBMISSION_DATE) + pd.DateOffset(days=7),
        y=max(sales_df["sales"]),
        text="Test Period",
        font=dict(size=14),
        ax=0,
        ay=0,
    )

    fig.update_layout(
        title="Sales Data with Holiday Events",
        xaxis_title="Date",
        yaxis_title="Sales",
        width=1200,
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


def get_oil_dcoilwtico_fig(oil_df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(20, 5))
    plt.plot(oil_df.index, oil_df["dcoilwtico"])

    plt.title("Oil Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Oil Price (dcoilwtico)")

    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig
