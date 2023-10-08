import streamlit as st
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from io import BytesIO
import fitz
import os
import time
import helper

URL = "https://data.bs.ch/api/explore/v2.1/catalog/datasets/{}/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B"
PARLIAMENT_DOCUMENTS = 100313


@st.cache_data
def get_ogd_dataset(id):
    df = pd.read_csv(URL.format(id), sep=";")
    return df


def download_pdf_files():
    def save_file(url, filename):
        response = requests.get(url, filename)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Failed to download PDF. Status code: {response.status_code}")

    df = pd.read_csv("./geschaeft.csv", sep=";")
    cnt = 1
    with st.empty():
        for index, row in df.iterrows():
            url = row["pdf_file_url"]
            parsed_url = urlparse(url)
            path = parsed_url.path
            filename = "./data/" + os.path.basename(path)
            if not os.path.exists(filename) and not url == "tbd":
                save_file(url, filename)
                st.write(f"Copied: {os.path.basename(path)} ({cnt}/{len(df)}))")
                # time.sleep(1)
            cnt += 1


def read_pdf_from_file(filename: str):
    # Step 1: Download the PDF
    with open(filename, "rb") as file:
        # Create a PDF reader object
        pdf_reader = fitz.PdfReader(file)
        # Initialize a variable to store PDF text
        pdf_text = ""
        # Loop through each page
        for page_num in range(len(pdf_reader.pages)):
            # Get a page object
            page = pdf_reader.pages[page_num]
            # Extract text from the page
            pdf_text += page.get_text()
    # Now, pdf_text contains the PDF content as a string
    return pdf_text


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


def main():
    df = get_ogd_dataset(PARLIAMENT_DOCUMENTS)
    st.dataframe(df.head())

    if st.button("download pdf files"):
        download_pdf_files()
    if st.button("extract texts"):
        for index, row in df.iterrows():
            if row["dok_laufnr"] != "ohne:":
                with open(f"./text_files/{row['dok_laufnr']}.txt", "w") as f:
                    try:
                        text = read_pdf_from_url(row["url_dok"])
                        if text:
                            f.write(text)
                    except Exception as e:
                        st.error(
                            f"document {row['dok_laufnr']} could not be written: {e}"
                        )


def prepare_votation():
    df = pd.read_csv("./data/100186.csv", sep=";")


if __name__ == "__main__":
    main()
