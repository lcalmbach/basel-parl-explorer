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


LOCAL_HOST = 'liestal'


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
    return os.environ[varname]
    # if socket.gethostname().lower() == LOCAL_HOST:
    #     return os.environ[varname]
    # else:
    #     return st.secrets[varname]
    
    
def add_year_date(df: pd.DataFrame, date_column: str, year_date_col: str):
    """
    Adds a year-end date to a DataFrame based on a specified date column.

    Args:
        df (pd.DataFrame): The DataFrame to add the year-end date to.
        date_column (str): The name of the column containing the date values.
        year_date_col (str): The name of the new column to add the year-end date to.

    Returns:
        pd.DataFrame: The original DataFrame with the new year-end date column added.
    """
    df[year_date_col] = df[date_column].dt.year.astype(str) + "-12-31"
    df[year_date_col] = pd.to_datetime(df[year_date_col])
    return df


def read_pdf_from_url(url: str):
    """
    Downloads a PDF from a given URL and returns its text content.

    Args:
        url (str): The URL of the PDF to download.

    Returns:
        str: The text content of the downloaded PDF.

    Raises:
        Exception: If the PDF fails to download or cannot be opened.
    """
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
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")


def show_download_button(df: pd.DataFrame, cfg: dict = {}):
    if "filename" not in cfg:
        cfg["filename"] = "file.csv"
    key = random_word(10)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=cfg["button_text"],
        data=csv,
        file_name=cfg["filename"],
        mime="text/csv",
        key=key,
    )


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


def random_word(length: int):
    """
    Generate a random string of lowercase letters with a given length.

    Args:
        length (int): The length of the random string to generate.

    Returns:
        str: A random string of lowercase letters with the given length.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def round_to_nearest(value, base):
    """
    Rounds the given value to the nearest multiple of the given base.

    Args:
        value (int): The value to be rounded.
        base (int): The base to which the value should be rounded.

    Returns:
        int: The rounded value.
    """
    return int(value / base / base) * base


LOCAL_HOST = "liestal"
