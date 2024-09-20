import warnings
import os

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from storesales.constants import REPORTS_PATH, EXTERNAL_TRAIN_PATH


def save_plot(
    store: int, family: str, store_family_data: pd.DataFrame, pdf: PdfPages
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(store_family_data["date"], store_family_data["sales"])
    plt.title(f'Store nbr: {store} - Family: "{family}"')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)

    pdf.savefig()
    plt.close()


def add_content_page(stores: list, pdf: PdfPages) -> None:
    """Adds the content page with links to each store."""
    plt.figure(figsize=(10, 6))
    plt.title("Table of Contents")
    plt.axis("off")  # Turn off axes

    content_text = "Table of Contents\n\n"
    for idx, store in enumerate(stores, start=1):
        content_text += f"{idx}. Store Number: {store}\n"

    plt.text(0.5, 0.5, content_text, ha="center", va="center", wrap=True, fontsize=12)
    pdf.savefig()
    plt.close()


def add_title_page(text: str, pdf: PdfPages) -> None:
    """Adds a title page"""
    plt.figure(figsize=(10, 6))

    plt.text(0.5, 0.5, text, color="black", fontsize=44, ha="center", va="center")

    plt.axis("off")
    pdf.savefig()
    plt.close()


def store_to_family_plots(
    df: pd.DataFrame, outer: list, inner: list, pdf: PdfPages, outer_store: bool
) -> None:
    store, family = None, None
    desc = "Processing Families"
    if outer_store:
        desc = "Processing Stores"

    for out in tqdm(outer, desc=desc):
        if outer_store:
            store = out
            add_title_page(f"Store № {store}", pdf)
        else:
            family = out
            add_title_page(f"{family}", pdf)

        for inn in inner:
            if outer_store:
                family = inn
            else:
                store = inn

            store_family_data = df[
                (df["store_nbr"] == store) & (df["family"] == family)
            ]

            if store_family_data.empty:
                warnings.warn(f"No data found for Store {store} - Family {family}")
                continue

            save_plot(store, family, store_family_data, pdf)


def main(outer_store: bool = True) -> None:
    df = pd.read_csv(EXTERNAL_TRAIN_PATH, parse_dates=["date"])
    pdf_file_path = os.path.join(REPORTS_PATH, "sales_plots.pdf")
    with PdfPages(pdf_file_path) as pdf:
        stores = sorted(df["store_nbr"].unique())
        families = df["family"].unique()

        if outer_store:
            outer = stores
            inner = families
        else:
            outer = families
            inner = stores

        # add_content_page(stores, pdf)
        store_to_family_plots(df, outer, inner, pdf, outer_store)


if __name__ == "__main__":
    main(outer_store=False)
