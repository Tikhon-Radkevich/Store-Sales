import os
import pandas as pd
import streamlit as st

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from explore_app_config import profile_kwargs
from dataset_utils.holidays import get_sales
from storesales.explore.utils import get_holidays_on_sales_fig, get_oil_dcoilwtico_fig
from storesales.constants import (
    EXTERNAL_DATA_PATH,
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_STORES_PATH,
    EXTERNAL_TRANSACTIONS_PATH,
    EXTERNAL_OIL_PATH,
    EXTERNAL_HOLIDAYS_EVENTS_PATH,
    EXTERNAL_SAMPLE_SUBMISSION_PATH,
)


class DataRenderer:
    @classmethod
    def render_data(cls, file_name: str) -> None:
        getattr(cls, f"render_{file_name}")()

    @staticmethod
    def render_holidays_events() -> None:
        train_df = DataLoader.load_train_data()
        holidays_df = DataLoader.load_holidays_events_data()

        slider_col, select_box_col = st.columns(2)
        with slider_col:
            store_nbr = st.slider("Store Number", 0, 54)
            if store_nbr == 0:
                store_nbr = None

        with select_box_col:
            family_options = ["All"] + train_df["family"].unique().tolist()
            family = st.selectbox("Family", family_options)
            if family == "All":
                family = None

        sales_df = get_sales(train_df, store_nbr, family)
        fig = get_holidays_on_sales_fig(sales_df, holidays_df)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_oil() -> None:
        oil_df = DataLoader.load_oil_data()
        fig = get_oil_dcoilwtico_fig(oil_df)
        st.pyplot(fig)

    @staticmethod
    def render_stores() -> None: ...

    @staticmethod
    def render_test() -> None: ...

    @staticmethod
    def render_train() -> None: ...

    @staticmethod
    def render_transactions() -> None: ...

    @staticmethod
    def render_sample_submission() -> None: ...


class DataLoader:
    dataset_files_paths = [
        file_path
        for file_path in os.listdir(EXTERNAL_DATA_PATH)
        if file_path.endswith(".csv")
    ]

    @classmethod
    def load_data(cls, file_name: str) -> pd.DataFrame:
        return getattr(cls, f"load_{file_name}_data")()

    @staticmethod
    @st.cache_data
    def load_train_data() -> pd.DataFrame:
        df = pd.read_csv(EXTERNAL_TRAIN_PATH, parse_dates=["date"])
        return df

    @staticmethod
    @st.cache_data
    def load_test_data() -> pd.DataFrame:
        df = pd.read_csv(EXTERNAL_TEST_PATH, parse_dates=["date"])
        return df

    @staticmethod
    @st.cache_data
    def load_stores_data() -> pd.DataFrame:
        return pd.read_csv(EXTERNAL_STORES_PATH)

    @staticmethod
    @st.cache_data
    def load_transactions_data() -> pd.DataFrame:
        df = pd.read_csv(EXTERNAL_TRANSACTIONS_PATH, parse_dates=["date"])
        return df

    @staticmethod
    @st.cache_data
    def load_oil_data() -> pd.DataFrame:
        oil_df = pd.read_csv(EXTERNAL_OIL_PATH, parse_dates=["date"])
        oil_df.set_index("date", inplace=True)
        oil_df = oil_df.asfreq("D")
        oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill()
        oil_df = oil_df.dropna()
        return oil_df

    @staticmethod
    @st.cache_data
    def load_holidays_events_data() -> pd.DataFrame:
        df = pd.read_csv(EXTERNAL_HOLIDAYS_EVENTS_PATH, parse_dates=["date"])
        return df

    @staticmethod
    @st.cache_data
    def load_sample_submission_data() -> pd.DataFrame:
        df = pd.read_csv(EXTERNAL_SAMPLE_SUBMISSION_PATH)
        return df


def render_profile_report(data: pd.DataFrame, file_name: str) -> None:
    profile_kwargs["title"] = f"Profile Report: {file_name}.csv"

    @st.cache_data
    def _get_profile_report(df: pd.DataFrame) -> ProfileReport:
        profile_cached = ProfileReport(df, explorative=True, **profile_kwargs)
        return profile_cached

    @st.cache_data
    def _get_profile_html(df: pd.DataFrame) -> str:
        profile_cached = ProfileReport(df, explorative=True, **profile_kwargs)
        return profile_cached.to_html()

    profile = _get_profile_report(data)
    profile_html = _get_profile_html(data)
    profile._html = profile_html
    st_profile_report(profile)
