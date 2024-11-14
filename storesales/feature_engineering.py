import pandas as pd


def create_date_features(df: pd.DataFrame, pref: str) -> pd.DataFrame:
    dates_dt = df["date"].dt

    date_features_df = pd.DataFrame(index=df.index)
    date_features_df[f"{pref}day"] = dates_dt.day
    date_features_df[f"{pref}month"] = dates_dt.month
    date_features_df[f"{pref}year"] = dates_dt.year
    date_features_df[f"{pref}day_of_week"] = dates_dt.dayofweek
    date_features_df[f"{pref}day_of_year"] = dates_dt.dayofyear
    return date_features_df
