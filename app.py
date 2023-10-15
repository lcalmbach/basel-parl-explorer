import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

from lang import get_used_languages, init_lang_dict_complete, get_lang
from grosser_rat import Parliament

__version__ = "0.0.6"
__author__ = "Lukas Calmbach"
__author_email__ = "lcalmbach@gmail.com"
VERSION_DATE = "2023-15-14"
APP_EMOJI = "🏛️"
APP_NAME = f"BaselParlExplorer"
GIT_REPO = ""

PAGE = "app"
LOTTIE_URL = "https://lottie.host/b3511e74-ab52-49a1-8105-7a6a49f00d4b/aDXvTKDGIX.json"


def init():
    st.set_page_config(  # Alternate names: setup_page, page, layout
        initial_sidebar_state="auto",
        page_title=APP_NAME,
        page_icon=APP_EMOJI,
        layout="wide",
    )
    if not ("lang" in st.session_state):
        init_lang_dict_complete("./lang/")
        # first item is default language
        st.session_state["used_languages_dict"] = get_used_languages(
            st.session_state["lang_dict"][PAGE]
        )
        st.session_state["lang"] = next(
            iter(st.session_state["used_languages_dict"].items())
        )[0]
        st.session_state.grosser_rat = Parliament()


def get_impressum():
    """
    Returns a string containing information about the application.
    Returns:
    - info (str): A formatted string containing details about the application.
    """
    created_by = lang["created_by"]
    powered_by = lang["powered_by"]
    version = lang["version"]
    translation = lang["text_translation"]

    info = f"""
    <div style="background-color:powderblue; padding: 10px;border-radius: 15px;">
    <small>{APP_EMOJI}{APP_NAME}<br>
    {created_by} <a href="mailto:{__author_email__}">{__author__}</a><br>
    {version}: {__version__} ({VERSION_DATE})<br>
    {powered_by} <a href="https://streamlit.io/">Streamlit</a> and 
    <a href="https://platform.openai.com/">OpenAI API</a><br> 
    <a href="{GIT_REPO}">git-repo</a><br> 
    {translation} <a href="https://lcalmbach-gpt-translate-app-i49g8c.streamlit.app/">PolyglotGPT</a>
    </small></div>
    """
    return info


def display_language_selection():
    """
    The display_info function displays information about the application. It
    uses the st.expander container to create an expandable section for the
    information. Inside the expander, displays the input and output format.
    """
    index = list(st.session_state["used_languages_dict"].keys()).index(
        st.session_state["lang"]
    )
    x = st.sidebar.selectbox(
        label=f'🌐{lang["language"]}',
        options=st.session_state["used_languages_dict"].keys(),
        format_func=lambda x: st.session_state["used_languages_dict"][x],
        index=index,
    )

    if x != st.session_state["lang"]:
        st.session_state["lang"] = x
        st.experimental_rerun()


@st.cache_data()
def get_lottie():
    """Performs a GET request to fetch JSON data from a specified URL.

    Returns:
        tuple: A tuple containing the JSON response and a flag indicating the
        success of the request.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the
        GET request. ValueError: If an error occurs while parsing the JSON
        response.
    """
    ok = True
    r = None
    try:
        response = requests.get(LOTTIE_URL)
        r = response.json()
    except requests.exceptions.RequestException as e:
        print(lang["get-request-error"]).format(e)
        ok = False
    except ValueError as e:
        print(lang["json-parsing-error"].format(e))
        ok = False
    return r, ok


def show_app_info():
    """
    The show_app_info function displays information about the application. It
    uses the st.expander container to create an expandable section for the
    information. Inside the expander, displays the input and output format.
    """
    cols = st.columns([1, 10, 1])
    with cols[1]:
        st.image("./info_app_wide.jpg")
        st.write("")
        st.header(f"{APP_EMOJI}{APP_NAME}")
        st.markdown(lang["app_info"], unsafe_allow_html=True)


def update_matters():
    from urllib.parse import urlparse

    df_matter = pd.read_csv("./matter_url.csv", sep=";")
    df_matter = df_matter[["signatur", "pdf_file_url"]]
    # Extract the path and split it to get the filename
    df_matter["key"] = df_matter["pdf_file_url"].apply(
        lambda x: urlparse(x).path.split("/")[-1]
    )
    df_summaries = pd.read_csv("./summaries.csv", sep="|")
    df_matter = df_matter.merge(df_summaries, on="key", how="left")
    df_matter.to_csv("./matter_url.csv", sep=";", index=False)


def main() -> None:
    """
    This function runs an app that classifies text data. Depending on the user's
    input option, it retrieves data from a demo or an uploaded file. Then,
    randomly selects a fixed number of records from the dataframe provided using
    record_selection function. The selected dataframe and dictionary of categories
    are previewed on the screen. If the user presses classify, the function runs the
    Classifier class on the selected dataframe, returns a response dataframe, and
    offers the user the option to download the dataframe to a CSV file.
    """
    global lang
    global grosser_rat

    init()
    update_matters()
    lang = get_lang(PAGE)
    lottie_search_names, ok = get_lottie()
    if ok:
        with st.sidebar:
            st_lottie(lottie_search_names, height=140, loop=20)
    else:
        pass

    menu_options = lang["menu_options"]
    # https://icons.getbootstrap.com/
    with st.sidebar:
        menu_action = option_menu(
            None,
            menu_options,
            icons=[
                "info-square",
                "people-fill",
                "collection-fill",
                "list",
                "hammer",
                "graph-up",
            ],
            menu_icon="cast",
            default_index=0,
        )

    if menu_action == menu_options[0]:
        show_app_info()
    elif menu_action == menu_options[1]:
        st.session_state.grosser_rat.members.select_item()
    elif menu_action == menu_options[2]:
        st.session_state.grosser_rat.bodies.select_item()
    elif menu_action == menu_options[3]:
        st.session_state.grosser_rat.pol_matters.select_item()
    elif menu_action == menu_options[4]:
        st.session_state.grosser_rat.votations.select_item()
    elif menu_action == menu_options[5]:
        st.session_state.grosser_rat.select_plot()
    display_language_selection()

    st.sidebar.markdown(get_impressum(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
