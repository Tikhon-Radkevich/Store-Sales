import os
import streamlit as st

from utils import DataLoader, DataRenderer, render_profile_report


def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Navigation")

    for file_path in DataLoader.dataset_files_paths:
        if st.sidebar.button(os.path.basename(file_path), use_container_width=True):
            st.session_state.selected_file = file_path

    selected_file = st.session_state.get("selected_file", None)
    if selected_file:
        file_name = str(os.path.basename(selected_file).removesuffix(".csv"))
        data = DataLoader.load_data(file_name)

        render_profile_report(data, file_name)

        DataRenderer.render_data(file_name)


if __name__ == "__main__":
    main()
