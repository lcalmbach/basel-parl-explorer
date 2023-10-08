import streamlit as st
import pandas as pd
import json
import os
import socket
import random
import string
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode
import fitz
import requests


def read_pdf_from_url(url: str):
    # Step 1: Download the PDF
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
            pdf = fitz.open("temp.pdf")
            text = ""
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text += page.get_text()
            return text
    else:
        st.error(f"Failed to download PDF. Status code: {response.status_code}")
        return "do document found"


def show_download_button(df: pd.DataFrame, cfg: dict = {}):
    if "filename" not in cfg:
        cfg["filename"] = "file.csv"
    key = randomword(10)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=cfg["button_text"],
        data=csv,
        file_name=cfg["filename"],
        mime="text/csv",
        key=key,
    )


def get_var(varname: str):
    if socket.gethostname().lower() == LOCAL_HOST:
        return os.environ[varname]
    else:
        return st.secrets[varname]


def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False


def get_var(varname: str) -> str:
    """
    Retrieves the value of a given environment variable or secret from the Streamlit configuration.

    If the current host is the local machine (according to the hostname), the environment variable is looked up in the system's environment variables.
    Otherwise, the secret value is fetched from Streamlit's secrets dictionary.

    Args:
        varname (str): The name of the environment variable or secret to retrieve.

    Returns:
        The value of the environment variable or secret, as a string.

    Raises:
        KeyError: If the environment variable or secret is not defined.
    """
    if socket.gethostname().lower() == LOCAL_HOST:
        return os.environ[varname]
    else:
        return st.secrets[varname]


def show_table(df: pd.DataFrame, cols=[], settings={}):
    def set_defaults():
        if "height" not in settings:
            settings["height"] = 400
        if "selection_mode" not in settings:
            settings["selection_mode"] = "single"
        if "fit_columns_on_grid_load" not in settings:
            settings["fit_columns_on_grid_load"] = True
        if "update_mode" not in settings:
            settings["update_mode"] = GridUpdateMode.SELECTION_CHANGED

    set_defaults()
    gb = GridOptionsBuilder.from_dataframe(df)
    # customize gridOptions
    gb.configure_default_column(
        groupable=False, value=True, enableRowGroup=False, aggFunc="sum", editable=False
    )
    for col in cols:
        gb.configure_column(
            col["name"],
            type=col["type"],
            precision=col["precision"],
            hide=col["hide"],
            width=col["width"],
        )
    gb.configure_selection(
        settings["selection_mode"], use_checkbox=False, rowMultiSelectWithClick=False
    )  # , suppressRowDeselection=suppressRowDeselection)
    gb.configure_grid_options(domLayout="normal")
    gridOptions = gb.build()
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        height=settings["height"],
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=settings["update_mode"],
        fit_columns_on_grid_load=settings["fit_columns_on_grid_load"],
        allow_unsafe_jscode=False,
        enable_enterprise_modules=False,
        # key=settings["key"],
    )
    selected = grid_response["selected_rows"]
    if selected:
        return selected[0]
    else:
        return []


def randomword(length: int):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


LOCAL_HOST = "liestal"
