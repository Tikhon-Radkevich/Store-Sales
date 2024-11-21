import argparse
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


def add_title_page(text: str, pdf: PdfPages) -> None:
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


def main(grouping_type: str) -> None:
    df = pd.read_csv(EXTERNAL_TRAIN_PATH, parse_dates=["date"])

    pdf_file_path = os.path.join(
        REPORTS_PATH,
        f"{'stores' if grouping_type == 'store' else 'families'}_plots.pdf",
    )

    with PdfPages(pdf_file_path) as pdf:
        stores = sorted(df["store_nbr"].unique())
        families = df["family"].unique()

        if grouping_type == "store":
            outer = stores
            inner = families
            add_title_page("Store-Based Grouping", pdf)
        elif grouping_type == "family":
            outer = families
            inner = stores
            add_title_page("Family-Based Grouping", pdf)
        else:
            raise ValueError("Invalid grouping type. Use 'store' or 'family'.")

        store_to_family_plots(df, outer, inner, pdf, grouping_type == "store")

    print(f"Plots saved to: {pdf_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sales plots grouped by store or family."
    )
    parser.add_argument(
        "grouping_type",
        type=str,
        choices=["store", "family"],
        help="Grouping type: 'store' to group by store, 'family' to group by family.",
    )
    args = parser.parse_args()

    main(args.grouping_type)
