import os
import streamlit as st

from utils import DataLoader, DataRenderer, render_profile_report


def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Navigation")
    selected_file = None

    for file_path in DataLoader.dataset_files_paths:
        if st.sidebar.button(os.path.basename(file_path)):
            selected_file = file_path

    if selected_file:
        file_name = os.path.basename(selected_file).removesuffix(".csv")
        data = DataLoader.load_data(file_name)

        st.title(f"Data Exploration: {file_name}")

        render_profile_report(data, file_name)

        DataRenderer.render_data(file_name)


if __name__ == "__main__":
    main()
