import streamlit as st
import json
import iso639
import os

# langdict: holds language tree
# > page: holds dict with all langunages for a page
# > page >lang holds the language dict for the page and language
# lang: holds current language


def init_lang_dict_complete(folder_path: str):
    """
    Retrieves the complete language dictionary from a JSON file.

    Returns:
    - lang (dict): A Python dictionary containing all the language strings.
    """
    st.session_state["lang_dict"] = {}
    for filename in os.listdir(folder_path):
        page = filename.split(".")[0]
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r") as file:
                    st.session_state["lang_dict"][page] = json.load(file)
            except FileNotFoundError:
                print("File not found.")
                return {}
            except json.JSONDecodeError:
                print("Invalid JSON format.")
                return {}
            except Exception as e:
                print("An error occurred:", str(e))
                return {}


def get_all_language_dict():
    """
    Retrieves a dictionary containing all the available languages and their
    ISO 639-1 codes.

    Returns:
        language_dict (dict): A Python dictionary where the keys are the ISO 639-1 codes and the values are the language names.
    """
    keys = [lang["iso639_1"] for lang in iso639.data if lang["iso639_1"] != ""]
    values = [lang["name"] for lang in iso639.data if lang["iso639_1"] != ""]
    language_dict = dict(zip(keys, values))
    return language_dict


def get_used_languages(lang_dict: dict):
    language_dict = get_all_language_dict()
    used_languages = list(lang_dict.keys())
    extracted_dict = {
        key: language_dict[key] for key in used_languages if key in language_dict
    }
    return extracted_dict


def get_lang(page: str) -> dict:
    """Retrieves the dictionary from the session statethe hierarchical
    organisation is lang_dict, then one key for ever py file (module)

    Args:
        page (str): every py file with multilang commands must have a file
                    with the same name and extension json in the lang folder

    Returns:
        _type_: _description_
    """
    try:
        return st.session_state["lang_dict"][page][st.session_state["lang"]]
    except KeyError as e:
        st.write(st.session_state["lang_dict"], e)
        return {}
    except Exception as e:
        print(st.session_state, e)
        return {}
