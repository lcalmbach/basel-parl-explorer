from typing import Any
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
from utils.lang import get_lang
from utils.helper import add_year_date, show_table, show_download_button
from enum import Enum
from utils.plots import time_series_line, bar_chart

# import boto3 # use when accessing text files on S3
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from whoosh.qparser import QueryParser

PAGE = __name__
MEMBER_COUNT = 100  # parliment members since 2008
DATA = "./data/"  # location of local data files
OGD_TABLE_URL = "https://data.bs.ch/api/explore/v2.1/catalog/datasets/{}/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B"
DEF_GRID_SETTING = {
    "fit_columns_on_grid_load": True,
    "height": 400,
    "selection_mode": "single",
}


# dataset key in data.bs.ch ogd portal
class Urls(Enum):
    MEMBERS_ALL = 100307
    INITIATIVES = 100086
    VOTATIONS = 100186
    MEMBERSHIPS = 100308
    DOCUMENTS = 100313
    BODIES = 100310


def lang(text):
    """
    Returns the translated text of the given text.
    It passes the PAGE variable holding the key to the current module

    Args:
        text (str): The text to determine the language of.

    Returns:
        str: translated text.
    """
    return get_lang(PAGE)[text]


def add_filter(filters: list, field: str, value: Any, operator: str = "eq"):
    """Adds a filter to the list of filters

    Args:
        filters (list): list of filters
        field (str): field to filter on
        value (Any): value to filter for
        operator (str, optional): filter operator. Defaults to "eq".

    Returns:
        list: updated list of filters
    """
    filters.append({"field": field, "value": value, "operator": operator})
    return filters


def filter(df: pd.DataFrame, filters: list):
    """Retrieves the filters fro ma list of filter object and applies them to the dataframe

    Args:
        filters (_type_): _description_

    Returns:
        pd.DataFrame:   dataframe with the filters applied
        list:           list of filter objects: eg: {"field": "Year", "operator": "eq", "value": 2020}
    """
    for filter in filters:
        # special case referring where a theme must be found in fields thema_1 and thema_2
        if filter["operator"] == "isin" and type(filter["field"] == list):
            df = df[
                (df["thema_1"].isin(filter["value"]))
                | (df["thema_2"].isin(filter["value"]))
            ]
        elif filter["operator"] == "eq":
            df = df[df[filter["field"]] == filter["value"]]
        elif filter["operator"] == "in":
            df = df[df[filter["field"]].isin(filter["value"])]
        elif filter["operator"] == "gt":
            df = df[df[filter["field"]] > filter["value"]]
        elif filter["operator"] == "st":
            df = df[df[filter["field"]] < filter["value"]]
        elif filter["operator"] == "contains":
            df = df[df[filter["field"]].str.contains(str(filter["value"]), case=False)]
        elif filter["operator"] == "isin":
            df = df[df[filter["field"]].isin(filter["value"], case=False)]
    return df


def get_table(table_nr: int):
    """
    Retrieves a table from the Basel-Stadt Open Data API and returns it as a
    pandas DataFrame.

    Args:
        table_nr (int): The ID of the table to retrieve.

    Returns:
        pandas.DataFrame: The retrieved table as a pandas DataFrame.
    """

    URL = OGD_TABLE_URL.format(table_nr)
    df = pd.read_csv(URL, sep=";")
    return df


class Documents:
    def __init__(self, parent, docs_df: pd.DataFrame, summaries: pd.DataFrame):
        """
        Initializes a Documents object.

        Args:
            parent: The parent object.
            docs_df: A pandas DataFrame containing the documents to be indexed.
        """
        self.parent = parent
        self.all_elements = self.get_elements(docs_df, summaries)
        # self.s3 = boto3.resource(
        #     "s3",
        #     aws_access_key_id=get_var("aws_access_key_id"),
        #     aws_secret_access_key=get_var("aws_secret_access_key"),
        #     region_name="us-east-1",
        # )

    def get_elements(self, df, summaries):
        """
        Gets the elements from a pandas DataFrame.

        Args:
            df: A pandas DataFrame.

        Returns:
            A pandas DataFrame containing the specified fields.
        """

        def transform_number(number):
            # Convert the number to a string
            number_str = str(number)
            num_zeros = 12 - len(number_str)
            transformed_number = "0" * num_zeros + number_str + ".pdf"
            return transformed_number

        fields = [
            "dokudatum",
            "dok_laufnr",
            "titel_dok",
            "laufnr_ges",
            "signatur_ges",
            "titel_ges",
            "url_ges",
            "url_dok",
            "key",
            "summary",
        ]
        df["signatur_ges"] = df["signatur_ges"].astype(str)
        df["dokudatum"] = pd.to_datetime(df["dokudatum"])
        df["year"] = df["dokudatum"].dt.year

        df["key"] = df["dok_laufnr"].apply(transform_number)
        df = df.merge(summaries, on="key", how="left")

        return df[fields]

    def filter(self, filters):
        """
        Filters the documents based on the specified filters.

        Args:
            filters: A list of filters.

        Returns:
            A pandas DataFrame containing the filtered documents.
        """
        for filter in filters:
            df = self.all_elements[
                self.all_elements[filter["field"]] == filter["value"]
            ]
        return df

    def create_index_aws(self):
        """
        Creates an index of documents stored in an S3 bucket using the Whoosh search engine.

        Returns:
            An index object representing the indexed documents.
        """
        print("creating index")
        # Define the schema for the index
        schema = Schema(filename=ID(stored=True), content=TEXT(stored=True))

        # Create an index writer
        ix = index.create_in("indexdir", schema)
        writer = ix.writer()
        # Initialize the S3 bucket
        bucket_name = "lc-opendata01"
        folder_name = "grosser_rat_bs_docs/"
        bucket = self.s3.Bucket(bucket_name)
        files = bucket.objects.filter(Prefix=folder_name)
        st.write(len(list(files)))
        # Loop through each file in the text_files folder and index them
        with st.spinner("Indexing documents..."):
            placeholder = st.empty()
            cnt = 1
            for obj in files:
                # reduce number to dok_nr
                dok_nr = obj.key.replace(".txt", "").replace(folder_name, "")
                placeholder.write(f"{dok_nr} ({cnt}/{len(list(files))})")
                content_object = self.s3.Object(bucket_name, obj.key)
                file_content = content_object.get()["Body"].read().decode("iso-8859-1")
                file_content = file_content.replace("\n", " ")
                writer.add_document(filename=dok_nr, content=file_content)
                cnt += 1
            writer.commit()
            return ix

    def search(self):
        """
        Searches the indexed documents using the Whoosh search engine.
        """
        searcher = self.ix.searcher()
        with st.columns(2)[0]:
            query_text = st.text_input(f"{lang('search_for')}:")
        if st.button("Search"):
            parser = QueryParser("content", self.ix.schema)
            query = parser.parse(query_text)
            results = searcher.search(query, limit=None)
            # Allow larger fragments
            results.fragmenter.maxchars = 300

            # Show more context before and after
            results.fragmenter.surround = 50
            # Display results
            st.write(lang("found_documents").format(len(results)))
            for hit in results:
                dok_laufnr = (
                    hit["filename"]
                    .replace(".txt", "")
                    .replace("grosser_rat_bs_docs/", "")
                )
                row = self.all_elements[
                    self.all_elements["dok_laufnr"] == int(dok_laufnr)
                ].iloc[0]
                st.write(
                    f"[**{row['titel_dok']}**]({row['url_dok']}) {lang('matter')}: [{row['signatur_ges']}]({row['url_ges']})"
                )
                st.markdown(
                    hit.highlights("content").replace("\n", " ") + "...",
                    unsafe_allow_html=True,
                )
            searcher.close()


class Body:
    def __init__(self, parent, row, id):
        self.parent = parent
        self.data = row
        self.name = row["name"]
        self.short_name = row["kurzname"]
        self.type = row["gremientyp"]
        self.id = id
        self.members = self.parent.bodies.get_members(self.id)
        self.current_president = self.parent.bodies.get_president(id)

    def show_detail(self):
        tabs = st.tabs([lang("info"), lang("members")])
        with tabs[0]:
            url = self.parent.bodies.url_dict.get(self.short_name)
            if url:
                st.markdown(f"**[🔗]({url}){self.name}**")
            else:
                st.markdown(f"**{self.name}**")

            series = self.data.T
            if not self.current_president.empty:
                series["aktive Mitglieder"] = self.parent.bodies.get_active_members(
                    self.id
                )
                partei = f"({self.current_president['partei_kname']})"
                series["präsident"] = (
                    f"{self.current_president['name_adr']} {self.current_president['vorname_adr']} "
                    + partei
                )

            st.markdown(pd.DataFrame(series).to_html(), unsafe_allow_html=True)
        with tabs[1]:
            cols = []
            st.markdown(
                f"**{lang('committee_members')} {self.name}: {len(self.members)}**"
            )
            show_table(self.members, cols, DEF_GRID_SETTING)


class Bodies:
    def __init__(self, parent, df_memberships, df_urls, df_bodies):
        self.parent = parent
        self.memberships = df_memberships
        self.all_elements = self.get_elements(df_bodies)
        self.filtered_elements = pd.DataFrame()
        self.url_dict = dict(zip(list(df_urls["key"]), list(df_urls["url"])))

    def get_filter(self):
        """Returns a list of filters to be applied on the members

        Args:
            df (pd.DataFrame): _description_
            filters (list): _description_

        Returns:
            _type_: _description_
        """
        filters = []
        with st.sidebar.expander(f"🔎{lang('filter')}", expanded=True):
            # political body name
            name = st.text_input(lang("body_name"))
            if name > "":
                filters = add_filter(filters, "name_gre", name, "contains")

            # political body type
            body_type_options = [lang("all")] + self.parent.body_type_options
            body_type = st.selectbox(lang("committee_type"), options=body_type_options)
            if body_type_options.index(body_type) > 0:
                filters = add_filter(filters, "gremientyp", body_type)

            # aktiv
            status_options = [lang("all"), lang("active"), lang("inactive")]
            status = st.selectbox(lang("active_inactive"), options=status_options)
            if status == lang("active"):
                filters = add_filter(filters, "aktiv", "Ja")
            elif status == lang("inactive"):
                filters = add_filter(filters, "aktiv", "Nein")
        return filters

    def get_elements(self, df):
        fields = ["ist_aktuelles_gremium", "kurzname", "name", "gremientyp", "uni_nr"]
        df = df[fields]
        df.columns = ["aktiv", "kurzname", "name", "gremientyp", "uni_nr"]
        return df

    def get_president(self, id):
        df = self.memberships[
            (self.memberships["uni_nr_gre"] == id)
            & (self.memberships["funktion_adr"].str.startswith("Präsident"))
        ]
        if len(df) > 0:
            df = df.sort_values(by="beginn_mit")
            return df.iloc[0]
        else:
            return pd.Series([])

    def get_active_members(self, id):
        df = self.memberships[
            (self.memberships["uni_nr_gre"] == id)
            & (self.memberships["ende_mit"].isna())
        ]
        count_members = len(df)
        return count_members

    def get_members(self, id):
        fields = [
            "name_adr",
            "vorname_adr",
            "partei_kname",
            "beginn_mit",
            "ende_mit",
            "funktion_adr",
            "uni_nr_adr",
            "uni_nr_gre",
        ]
        df = self.memberships[fields][self.memberships["uni_nr_gre"] == id]
        df["aktiv"] = df["ende_mit"].apply(lambda x: "Ja" if pd.isnull(x) else "Nein")
        return df

    def select_item(self):
        filters = self.get_filter()
        self.filtered_elements = filter(self.all_elements, filters)

        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "field": "bodies",
        }
        cols = [
            {
                "name": "uni_nr",
                "type": "int",
                "precision": None,
                "hide": True,
                "width": None,
                "format": None,
            },
        ]
        st.subheader(
            f"{lang('parliament_committees')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        st.markdown(lang("body_table_help"))
        sel_member = show_table(self.filtered_elements, cols, settings)
        cfg = {"filename": f"{lang('bodies')}.txt", "button_text": lang("download")}
        show_download_button(self.filtered_elements, cfg)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("uni_nr").loc[sel_member["uni_nr"]]
            self.current_body = Body(self.parent, row, sel_member["uni_nr"])
            self.current_body.show_detail()


class Votation:
    def __init__(self, parent, row, id):
        self.parent = parent
        self.data = row
        self.results = self.parent.votations.filter_results(
            [
                {"field": "id_geschaeft", "value": id},
            ]
        )
        self.id = id
        self.date = row["Datum"]
        self.matter = row["Geschaeft"]
        self.yes = row["Ja-Stimmen"]
        self.nos = row["Nein-Stimmen"]
        self.abstentions = row["Enthaltungen"]
        self.absentees = row["Abwesende"]
        self.casting_votes = row["Präsidiumsstimmen"]

    def show_detail(self):
        fields = ["Name", "Fraktion", "Entscheid Mitglied"]
        df_j = self.results[self.results["Entscheid Mitglied"] == "J"][fields]
        df_n = self.results[self.results["Entscheid Mitglied"] == "N"][fields]
        df_e = self.results[self.results["Entscheid Mitglied"] == "E"][fields]
        df_a = self.results[self.results["Entscheid Mitglied"] == "A"][fields]
        df_p = self.results[self.results["Entscheid Mitglied"] == "P"][fields]
        tabs = st.tabs(
            [
                f"{lang('yeas')} ({len(df_j)})",
                f"{lang('nays')} ({len(df_n)})",
                f"{lang('abstentions')} ({len(df_e)})",
                f"{lang('absences')} ({len(df_a)})",
                f"{lang('presidium_votes')} ({len(df_p)})",
            ]
        )
        with tabs[0]:
            st.dataframe(df_j, hide_index=True)
        with tabs[1]:
            st.dataframe(df_n, hide_index=True)
        with tabs[2]:
            st.dataframe(df_e, hide_index=True)
        with tabs[3]:
            st.dataframe(df_a, hide_index=True)
        with tabs[4]:
            st.dataframe(df_p, hide_index=True)


class Votations:
    def __init__(self, parent, df_matters, df_results):
        self.parent = parent
        df_results["Datum"] = pd.to_datetime(df_results["Datum"])
        self.results = df_results
        df_results = add_year_date(df_results, "Datum", "jahr_datum")

        self.all_elements = self.get_elements(df_matters)
        self.all_elements["datum_fmt"] = self.all_elements["Datum"].dt.strftime(
            "%Y-%m-%d"
        )
        self.date_options = sorted(
            self.all_elements["datum_fmt"].unique(), reverse=True
        )
        self.first_year = self.all_elements["jahr"].min()
        self.last_year = self.all_elements["jahr"].max()
        self.year_options = sorted(
            list(range(self.first_year, self.last_year + 1)), reverse=True
        )
        self.type_options = sorted(list(self.all_elements["Typ"].unique()))
        self.filtered_elements = pd.DataFrame()
        # results by fraction and restult type
        self.settings_plot = {
            "x": "jahr_datum",
            "x_title": lang("year"),
            "x_format": "%Y",
            "y": "cnt_p_member",
            "y_title": "Anzahl pro Mitglied",
            "tooltip": ["jahr_datum", "cnt_p_member"],
            "width": 800,
            "height": 600,
            "groups": ["SP", "SVP", "FDP", "LDP", "GLP"],
            "result": "J",
        }

    def get_filter(self):
        """Returns a list of filters to be applied on the members

        Args:
            df (pd.DataFrame): _description_
            filters (list): _description_

        Returns:
            _type_: _description_
        """
        filters = []
        with st.sidebar.expander(f"🔎{lang('filter')}", expanded=True):
            # Geschäft
            matter = st.text_input(lang("matter"))
            if matter > "":
                filters = add_filter(filters, "Geschaeft", matter, "contains")

            # election year
            year_options = [lang("all")] + list(self.year_options)
            year = st.selectbox(lang("year"), year_options)
            if year_options.index(year) > 0:
                filters = add_filter(filters, "jahr", year)

            # date
            date_options = [lang("all")] + list(self.date_options)
            date = st.selectbox(lang("votation_date"), date_options)
            if date_options.index(date) > 0:
                filters = add_filter(filters, "datum_fmt", date)

            # result
            result_options = [lang("all"), lang("accepted"), lang("rejected")]
            result = st.selectbox(lang("result"), result_options)
            if result == lang("accepted"):
                filters = add_filter(filters, "Angenommen", lang("yes"))
            elif result == lang("rejected"):
                filters = add_filter(filters, "Angenommen", lang("no"))

            # vote type
            type_options = [lang("all")] + self.type_options
            type = st.selectbox(lang("type"), type_options)
            if type_options.index(type) > 0:
                filters = add_filter(filters, "Typ", type)

        return filters

    def filter_results(self, filters, add_details=False):

        for filter in filters:
            df = self.results[self.results[filter["field"]] == filter["value"]]
        if add_details:
            df = df.merge(self.all_elements, left_on="id_geschaeft", right_on="id")
        return df

    def get_elements(self, df):
        df["Datum"] = pd.to_datetime(df["Datum"])
        df["jahr"] = df["Datum"].dt.year
        df = add_year_date(df, "Datum", "jahr_datum")
        df["pzt_ja"] = df["Ja-Stimmen"] / (MEMBER_COUNT - df["Abwesende"]) * 100.0
        df["pzt_abwesend"] = df["Abwesende"] / MEMBER_COUNT * 100
        df["Angenommen"] = df["Ja-Stimmen"] > df["Nein-Stimmen"]
        df["Angenommen"] = df.apply(
            lambda row: lang("yes")
            if row["Ja-Stimmen"] > row["Nein-Stimmen"]
            else lang("no"),
            axis=1,
        )
        return df

    def get_member_votes(self, id):
        fields = [
            "Abstimmungsnummer",
            "Sitz Nr.",
            "Name",
            "Fraktion",
            "Entscheid Mitglied",
            "Zeitstempel",
        ]
        df = self.results[fields][self.results["Abstimmungsnummer"] == id]
        return df

    def select_item(self):
        filters = ["votation_year"]
        filters = self.get_filter()
        self.filtered_elements = filter(self.all_elements, filters)
        fields = [
            "datum_fmt",
            "Geschaeft",
            "Abstimmungsnummer",
            "Typ",
            "Angenommen",
            "Ja-Stimmen",
            "Nein-Stimmen",
            "Enthaltungen",
            "Abwesende",
            "Präsidiumsstimmen",
            "id",
        ]
        settings = {
            "fit_columns_on_grid_load": True,
            "height": 400,
            "selection_mode": "single",
            "field": "bodies",
        }
        df = self.filtered_elements[fields]
        df.columns = [
            "Datum",
            "Geschäft",
            "Abstimmungsnummer",
            "Typ",
            "Angenommen",
            "Ja",
            "Nein",
            "Enthaltungen",
            "Abwesende",
            "Präsidiumsstimmen",
            "id",
        ]
        cols = [
            {
                "name": "Datum",
                "type": "date",
                "precision": None,
                "hide": False,
                "width": None,
                "format": "%Y-&m-%d",
            },
            {
                "name": "id",
                "type": "int",
                "precision": None,
                "hide": True,
                "width": 0,
                "format": None,
            },
        ]
        df = df.sort_values(by=df.columns[0], ascending=False)
        st.subheader(f"{lang('votations')} ({len(df)}/{len(self.all_elements)})")
        st.markdown(lang("votations_table_help"))
        sel_member = show_table(df, cols, settings)
        cfg = {
            "filename": f"{lang('votations_short')}.txt",
            "button_text": lang("download"),
        }
        show_download_button(df, cfg)
        if len(sel_member) > 0:
            row = self.filtered_elements[
                self.filtered_elements["id"] == sel_member["id"]
            ].iloc[0]
            self.current_body = Votation(self.parent, row, sel_member["id"])
            self.current_body.show_detail()

    def show_plot(self):
        title_dict = {
            "J": lang("no_yeas"),
            "N": lang("no_nays"),
            "E": lang("no_abstentions"),
            "A": lang("absences"),
            "P": lang("no_presidium_votes"),
        }
        results_dict = {
            "J": lang("yeas"),
            "N": lang("nays"),
            "E": lang("abstentions"),
            "A": lang("absences"),
            "P": lang("presidium_votes"),
        }

        def get_members_per_fraction():
            group_fields = ["Jahr", "jahr_datum", "Fraktion"]
            df = self.results[group_fields + ["Sitz Nr."]].drop_duplicates()
            df = (
                df[group_fields]
                .groupby(["Jahr", "jahr_datum", "Fraktion"])
                .size()
                .reset_index(name="Members")
            )
            return df

        def show_settings():
            cols = st.columns(2)
            old_groups, old_result = (
                self.settings_plot["groups"],
                self.settings_plot["result"],
            )
            with cols[0]:
                group_options = self.results["Fraktion"].unique()
                group_options = [x for x in group_options if x != "="]
                self.settings_plot["groups"] = st.multiselect(
                    label=lang("pol_groups"),
                    options=group_options,
                    default=self.settings_plot["groups"],
                )
                result_options = list(results_dict.keys())
                self.settings_plot["result"] = st.selectbox(
                    label=f"{lang('decision')}/{lang('absence')}",
                    options=result_options,
                    format_func=lambda x: results_dict[x],
                    index=result_options.index(self.settings_plot["result"]),
                )
                # if there are changes, rerun the app
                if (old_groups != self.settings_plot["groups"]) | (
                    old_result != self.settings_plot["result"]
                ):
                    st.rerun()

        with st.sidebar:
            df_fract_members = get_members_per_fraction()
            group_fields = ["Jahr", "jahr_datum", "Fraktion"]
            df = self.results[
                self.results["Entscheid Mitglied"] == self.settings_plot["result"]
            ]

            if len(self.settings_plot["groups"]) > 0:
                df = df[df["Fraktion"].isin(self.settings_plot["groups"])]
            df = df[group_fields].groupby(group_fields).size().reset_index(name="Count")
            df = df.merge(df_fract_members, on=group_fields)
            df["cnt_p_member"] = df["Count"] / df["Members"]
            self.settings_plot["color"] = "Fraktion"
            self.settings_plot[
                "title"
            ] = f"{title_dict[self.settings_plot['result']]} {lang('per_member_and_fraction')}"
        self.settings_plot["y_domain"] = [0, df["cnt_p_member"].max()]
        # self.settings_plot["x_domain"] = [
        #   int(df["jahr_datum"].min()),
        #   int(df["jahr_datum"].max()),
        # ]
        self.settings_plot["tooltip"] = [
            "Jahr",
            self.settings_plot["y"],
            self.settings_plot["color"],
        ]

        tabs = st.tabs([lang("graph"), lang("settings")])

        with tabs[1]:
            show_settings()
        with tabs[0]:
            time_series_line(df, self.settings_plot)
            text = (
                lang("figure0_legend").format(title_dict[self.settings_plot["result"]])
                + " "
                + lang("explain_absences_plot")
            )
            st.markdown(text)


class Member:
    def __init__(self, parent, row, id):
        self.parent = parent
        if type(row) == pd.DataFrame:
            row = row.sort_values(by="gr_beginn", ascending=False).iloc[0]
        self.data = pd.DataFrame(row).T.reset_index()
        self.name = row["name"]
        self.first_name = row["vorname"]
        self.place_id = row["gr_sitzplatz"]
        self.party = row["partei_kname"]
        self.uni_nr = id
        self.url = row["url"]
        self.is_active_member = row["active_member"]

        self.full_name = f"{self.first_name} {self.name}"
        self.pol_matters = filter(
            self.parent.pol_matters.all_elements,
            [
                {
                    "field": "urheber",
                    "operator": "eq",
                    "value": f"{self.name}, {self.first_name}",
                }
            ],
        )
        self.memberships = self.parent.memberships.filter(
            [{"field": "uni_nr_adr", "value": self.uni_nr}]
        )
        self.votations = self.parent.votations.filter_results(
            [
                {"field": "Name", "value": self.full_name},
            ],
            add_details=True,
        )
        self.votation_meetings_no = len(self.votations["datum_fmt"].unique())
        self.votation_num = len(self.votations)
        self.votation_yes = len(
            self.votations[self.votations["Entscheid Mitglied"] == "J"]
        )
        self.votation_no = len(
            self.votations[self.votations["Entscheid Mitglied"] == "N"]
        )
        self.votation_abs = len(
            self.votations[self.votations["Entscheid Mitglied"] == "A"]
        )
        self.votation_abst = len(
            self.votations[self.votations["Entscheid Mitglied"] == "E"]
        )

    def show_detail(self):
        if self.url != np.nan:
            st.markdown(f"**[🔗]({self.url}){self.full_name}**", unsafe_allow_html=True)
        else:
            st.markdown(f"**{self.full_name}**", unsafe_allow_html=True)

        tabs = st.tabs(lang("member_tabs"))
        with tabs[0]:
            fields = [
                "index",
                "ist_aktuell_grossrat",
                "titel",
                "anrede",
                "name",
                "vorname",
                "gebdatum",
                "gr_wahlkreis",
                "partei_kname",
                "wahljahr",
                "austritt",
                "gr_beruf",
                "gr_arbeitgeber",
                "homepage",
            ]
            df = self.data[fields].copy()
            if df["austritt"].iloc[0]:
                df["austritt"] = df["austritt"].astype(int)
            df = df.T.reset_index()
            df.columns = lang("field_value_cols")
            df.dropna(inplace=True)
            if self.votation_num > 0:
                df_votations = pd.DataFrame(
                    {
                        "Feld": [
                            lang("votes_possible"),
                            lang("votes_yes"),
                            lang("votes_no"),
                            lang("votes_abstention"),
                            lang("votes_absent"),
                        ],
                        "Wert": [
                            self.votation_num,
                            f"{self.votation_yes}  ({self.votation_yes/self.votation_num*100:.0f}%)",
                            f"{self.votation_no} ({self.votation_no/self.votation_num*100:.0f}%)",
                            f"{self.votation_abst} ({self.votation_abst/self.votation_num*100:.0f}%)",
                            f"{self.votation_abs} ({self.votation_abs/self.votation_num*100:.0f}%)",
                        ],
                    }
                )
                df = pd.concat((df, df_votations))
            table = df.to_html(index=False)

            st.markdown(table, unsafe_allow_html=True)
        with tabs[1]:
            fields = ["signatur", "geschaftstyp", "titel", "beginn_datum", "ende"]
            cols = []
            self_matter = show_table(self.pol_matters[fields], cols, DEF_GRID_SETTING)
            if len(self_matter) > 0:
                row = self.pol_matters[
                    self.pol_matters["signatur"] == self_matter["signatur"]
                ].iloc[0]
                pol_matter = PolMatter(self.parent, row, self_matter["signatur"])
                pol_matter.show_detail()
        with tabs[2]:
            fields = [
                "kurzname_gre",
                "name_gre",
                "gremientyp",
                "beginn_mit",
                "ende_mit",
                "funktion_adr",
                "uni_nr_gre",
            ]
            cols = []
            sel_membership = show_table(
                self.memberships[fields], cols, DEF_GRID_SETTING
            )
            if len(sel_membership) > 0:
                row = self.memberships[
                    self.memberships["uni_nr_gre"] == sel_membership["uni_nr_gre"]
                ].iloc[0]
                matter = PolMatter(self.parent, row)
                matter.show_detail()
        with tabs[3]:
            fields = [
                "Datum_x",  # from merge
                "Geschaeft",
                "Typ",
                "Abstimmungsnummer",
                "Entscheid Mitglied",
            ]
            df = self.votations[fields]
            df.rename(
                columns={"Datum_x": "Datum", "Geschaeft": "Geschäft"}, inplace=True
            )
            cols = []
            if len(df) > 0:
                show_table(df, cols, DEF_GRID_SETTING)
            else:
                st.markdown(lang("no_votations"))


class Parties:
    def __init__(self, parent, df_seats):
        self.parent = parent
        self.all_elements = self.get_elements(df_seats)
        self.filtered_elements = pd.DataFrame()
        self.settings_plot = {
            "x": lang("seats"),
            "x_dt": "Q",
            "x_title": lang("year"),
            "y": "Jahr",
            "y_dt": "O",
            "y_title": lang("year"),
            "width": 800,
            "height": 1000,
            "color": "Partei",
            "color_scheme": {
                "PdA, KP": "darkred",
                "GB, FraB, POB": "green",
                "SP": "red",
                "GLP": "lightgreen",
                "EVP, VEW": "yellow",
                "CVP, KVP": "orange",
                "LDP, LP": "darkblue",
                "LdU": "darkorange",
                "FDP, RDP": "blue",
                "SVP, BGP": "darkgreen",
                "VA, SD, UVP, NA": "brown",
                "DSP": "pink",
                "Andere": "grey",
            },
            "bar_width": 18,
            "tooltip": ["Jahr", "Partei", lang("seats")],
        }
        self.party_url_dict = {
            "AB": "https://https://www.aktivesbettingen.ch/",
            "BastA!": "https://basta-bs.ch/",
            "EVP": "https://www.evp-bs.ch/",
            "SP": "https://www.sp-bs.ch/",
            "LDP": "https://ldp.ch/",
            "SVP": "https://www.svp-basel.ch/",
            "GLP": "https://bs.grunliberale.ch/",
            "FDP": "https://www.fdp-bs.ch/",
            "Grüne": " https://gruene-bs.ch/",
            "Mitte": " https://bs.die-mitte.ch/",
            "jgb": "https://www.jungesgruenesbuendnis.ch/",
            "VA": "http://www.ericweber.net/de/?id=startseite",
        }
        # make sure the parties appear in the right order,
        # see: https://de.wikipedia.org/wiki/Grosser_Rat_(Basel-Stadt)
        self.settings_plot["x_sort"] = list(self.settings_plot["color_scheme"].keys())

    def get_filter(self):
        """Returns a list of filters to be applied on the members

        Args:
            df (pd.DataFrame): _description_
            filters (list): _description_

        Returns:
            _type_: _description_
        """
        filters = []
        with st.sidebar.expander(f"🔎{lang('filter')}", expanded=True):
            ...

        return filters

    def get_elements(self, df_seats):
        """
        This function takes a DataFrame of seat counts and returns a melted DataFrame with the year, party, and count columns.

        Args:
        - df_seats: A pandas DataFrame containing seat counts for each party and year.

        Returns:
        - A pandas DataFrame with the year, party, and count columns.
        """
        melted_df = pd.melt(
            df_seats, id_vars=["Jahr"], var_name="Partei", value_name=lang("seats")
        )
        return melted_df

    def select_item(self):
        ...

    def show_plot(self):
        party_sort = {
            "PdA, KP": 1,
            "GB, FraB, POB": 2,
            "SP": 3,
            "DSP": 4,
            "GLP": 5,
            "EVP, VEW": 6,
            "CVP, KVP": 7,
            "LDP, LP": 8,
            "LdU": 9,
            "FDP, RDP": 10,
            "SVP, BGP": 11,
            "VA, SD, UVP, NA": 12,
            "Andere": 13,
        }

        df = self.all_elements.copy()
        df["sort_key"] = df["Partei"].map(party_sort)
        options_show_parties = [lang("seats"), lang("percent_seats")]
        with st.columns(2)[0]:
            show_pct = st.selectbox("Darstellung", options_show_parties)
        if options_show_parties.index(show_pct) == 1:
            mask = df["Jahr"] < 2008
            df.loc[mask, lang("seats")] = (df.loc[mask, lang("seats")] / 130) * 100
            df.loc[mask, lang("seats")] = df.loc[mask, lang("seats")].round(2)
            self.settings_plot["x_title"] = lang("percent_seats")
            self.settings_plot["x_domain"] = [0, 100]

        bar_chart(df, self.settings_plot)
        st.markdown(lang("seat_distribution_legend"))
        st.markdown(f"*{lang('parties')}*:")
        text = ""
        parties = sorted(list(self.parent.parties_dict.keys()))
        for party in parties:
            text += f"- [{party}]({self.party_url_dict[party]}): {self.parent.parties_dict[party]}\n"
        st.markdown(text)


class Members:
    def __init__(self, parent, df_members):
        self.parent = parent
        self.all_elements = self.get_elements(df_members)
        self.filtered_elements = pd.DataFrame()

    def get_filter(self):
        """Returns a list of filters to be applied on the members

        Args:
            df (pd.DataFrame): _description_
            filters (list): _description_

        Returns:
            _type_: _description_
        """
        filters = []
        with st.sidebar.expander(f"🔎{lang('filter')}", expanded=True):
            # name
            name = st.text_input(lang("member_name"))
            if name > "":
                filters = add_filter(filters, "name", name, "contains")

            # election year
            year_options = self.parent.year_options
            year_options = [lang("all")] + list(year_options)
            year = st.selectbox(lang("election_year"), year_options)
            if year_options.index(year) > 0:
                filters = add_filter(filters, "wahljahr", year)

            # member is active
            member_options = [
                lang("all"),
                lang("active_members"),
                lang("former_members"),
            ]
            is_member = st.selectbox(lang("membership_status"), options=member_options)
            if member_options.index(is_member) == 1:
                filters = add_filter(filters, "active_member", True)
            elif member_options.index(is_member) == 2:
                filters = add_filter(filters, "active_member", False)

            # electoral district
            district_options = [lang("all")] + self.parent.electoral_district_options
            district = st.selectbox(lang("voting_district"), district_options)
            if district_options.index(district) > 0:
                filters = add_filter(filters, "gr_wahlkreis", district)

            # gender
            gender_options = [lang("all")] + ["M", "F"]
            gender = st.selectbox(lang("gender"), gender_options)
            if gender_options.index(gender) > 0:
                filters = add_filter(filters, "geschlecht", gender)

        return filters

    def get_elements(self, df_members):
        fields = [
            "ist_aktuell_grossrat",
            "anrede",
            "titel",
            "name",
            "vorname",
            "gebdatum",
            "gr_sitzplatz",
            "gr_wahlkreis",
            "partei_kname",
            "gr_beginn",
            "gr_ende",
            "uni_nr",
            "strasse",
            "plz",
            "ort",
            "gr_beruf",
            "gr_arbeitgeber",
            "homepage",
            "url",
            "url_gremiumsmitgliedschaften",
            "url_interessensbindungen",
            "url_urheber",
        ]

        df = df_members[fields]
        df = df.assign(gr_beginn=pd.to_datetime(df["gr_beginn"]))
        df = df.assign(gr_ende=pd.to_datetime(df["gr_ende"]))
        df = df.assign(wahljahr=df["gr_beginn"].dt.year.fillna(0).astype(int))
        df = df.assign(austritt=df["gr_ende"].dt.year.fillna(0).astype(int))
        df = df.assign(gr_sitzplatz=df["gr_sitzplatz"].fillna(0).astype(int))
        df = df.assign(geschlecht=["M" if x == "Herr" else "F" for x in df["anrede"]])
        df = df.assign(
            active_member=[
                True if x == "Ja" else False for x in df["ist_aktuell_grossrat"]
            ]
        )
        return df

    def select_item(self):
        filters = self.get_filter()
        self.filtered_elements = filter(self.all_elements, filters)
        fields = [
            "name",
            "vorname",
            "ist_aktuell_grossrat",
            "gr_wahlkreis",
            "partei_kname",
            "wahljahr",
            "austritt",
            "uni_nr",
        ]
        settings = {
            "height": 400,
            "selection_mode": "single",
            "field": "members",
        }
        cols = []
        st.subheader(
            f"{lang('parliament_members')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        st.markdown(lang("member_table_help"))
        df = self.filtered_elements[fields].copy()
        sel_member = show_table(df, cols, settings)
        cfg = {"filename": f"{lang('members')}.txt", "button_text": lang("download")}
        show_download_button(df, cfg)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("uni_nr").loc[sel_member["uni_nr"]]
            self.current_member = Member(self.parent, row, sel_member["uni_nr"])
            self.current_member.show_detail()


class PolMatter:
    def __init__(self, parent, row, signatur):
        self.parent = parent
        self.data = pd.DataFrame(row).T.reset_index()
        self.data["themen"] = self.data["thema_1"]
        self.data["themen"][self.data["thema_2"].notna()] = (
            self.data["thema_1"] + ", " + self.data["thema_2"]
        )
        self.signatur = signatur
        self.type = row["geschaftstyp"]
        self.urheber = row["urheber"]
        self.title = row["titel"]
        self.start_date = row["beginn_datum"]
        self.end_date = row["ende"]
        self.url_matter = row["geschaft"]
        self.url_doc = row["pdf_file_url"]

        self.documents = self.parent.documents.filter(
            [{"field": "signatur_ges", "value": signatur}]
        ).sort_values(by="dokudatum")
        self.summary = self.documents.iloc[0]["summary"]

    def show_detail(self):
        st.markdown(f"{lang('matter')}: **{self.title}**")
        tabs = st.tabs(lang("matter_tabs"))
        with tabs[0]:
            fields = [
                "titel",
                "geschaftstyp",
                "urheber",
                "partei",
                "beginn_datum",
                "ende",
                "themen",
                "status",
                "summary",
            ]
            df = self.data[fields].copy()
            df.iloc[0]["summary"] = self.summary
            df["status"] = [
                lang("in_progress") if x == "B" else lang("closed")
                for x in df["status"]
            ]
            df.columns = [
                "Titel",
                "Geschäftstyp",
                "Urheberperson",
                "Partei",
                "Beginn",
                "Ende",
                "Themen",
                "Status",
                "Zusammenfassung",
            ]
            df = pd.DataFrame(df.T).reset_index().dropna()
            df.columns = lang("field_value_cols")
            table = df.to_html(index=False)
            st.markdown(table, unsafe_allow_html=True)
            st.write("")
            st.markdown(
                f"<sub>{lang('summary_ki_generated')}</sub>", unsafe_allow_html=True
            )
            st.link_button(
                f"🔗{lang('matter')}", self.url_matter, help=lang("more_info_pol_matter")
            )
        with tabs[1]:
            text = ""
            for index, row in self.documents.iterrows():
                text += f"1. {row['dokudatum'].strftime('%d.%m.%y')} [{row['titel_dok']}]({row['url_dok']})\n"
            st.markdown(text)


class PolMatters:
    def __init__(self, parent, df):
        self.parent = parent
        self.all_elements = self.get_elements(df)
        self.first_year = self.all_elements["beginn_datum"].dt.year.min()
        self.last_year = self.all_elements["beginn_datum"].dt.year.max()
        self.year_options = sorted(
            list(range(self.first_year, self.last_year + 1)), reverse=True
        )
        self.theme_options = self.get_themes(self.all_elements)
        self.settings_plot = {
            "x": "jahr_datum",
            "x_title": lang("year"),
            "x_format": "%Y",
            "y": "count",
            "y_title": lang("figure1_ytitle"),
            "width": 800,
            "height": 600,
            "themes": [],
            "color": None,
        }

    def get_themes(self, df: pd.DataFrame) -> list:
        concat_list = list(df["thema_1"].unique()) + list(df["thema_2"].unique())
        concat_list = [x for x in concat_list if x is not np.nan]
        concat_list = list(set(concat_list))
        return sorted(concat_list)

    def get_filter(self, excluded_list: list = []):
        """Returns a list of filters to be applied on the members

        Args:
            df (pd.DataFrame): _description_
            filters (list): _description_

        Returns:
            _type_: _description_
        """
        # ["matter_title", "year", "author", "theme"]
        filters = []
        with st.sidebar.expander(f"🔎{lang('filter')}", expanded=True):
            # political body name
            if "matter_title" not in excluded_list:
                name = st.text_input(lang("matter"))
                if name > "":
                    filters = add_filter(filters, "titel", name, "contains")

            # signatur
            if "signatur" not in excluded_list:
                signatur = st.text_input("Signatur")
                if signatur > "":
                    filters = add_filter(filters, "signatur", signatur, "contains")

            # year
            if "year" not in excluded_list:
                year_options = [lang("all")] + self.year_options
                year = st.selectbox(lang("year"), options=year_options)
                if year_options.index(year) > 0:
                    filters = add_filter(filters, "jahr", year)

            # matter type
            if "type" not in excluded_list:
                type_options = [lang("all")] + self.parent.matter_options
                matter_type = st.selectbox(lang("matter_type"), options=type_options)
                if type_options.index(matter_type) > 0:
                    filters = add_filter(filters, "geschaftstyp", matter_type)

            # political party
            if "party" not in excluded_list:
                party_options = [lang("all")] + self.parent.party_options
                party = st.selectbox(lang("party"), options=party_options)
                if party_options.index(party) > 0:
                    filters = add_filter(filters, "partei", party)

            # theme
            if "theme" not in excluded_list:
                themes = st.multiselect(
                    lang("theme"),
                    self.theme_options,
                    help="Selektiere ein oder mehrere Themen",
                )
                self.settings_plot["themes"] = themes
                if len(themes) > 0:
                    filters = add_filter(
                        filters, ["thema_1", "thema_2"], themes, "isin"
                    )

            # Status
            if "status" not in excluded_list:
                status_options = [lang("all"), lang("in_progress"), lang("closed")]
                status = st.selectbox(lang("status"), options=status_options)
                if status_options.index(status) == 1:
                    filters = add_filter(filters, "status", "B")
                elif status_options.index(status) == 2:
                    filters = add_filter(filters, "status", "A")

        return filters

    def get_elements(self, df):
        df_urls = pd.read_csv(DATA + "matter_url.csv", sep=";")
        df["signatur"] = df["signatur"].astype(str)
        df_urls["signatur"] = df_urls["signatur"].astype(str)
        df = df.merge(df_urls, on="signatur", how="left")
        df["beginn_datum"] = pd.to_datetime(df["beginn_datum"])
        df = add_year_date(df, "beginn_datum", "jahr_datum")
        df["jahr"] = df["beginn_datum"].dt.year.astype(int)

        return df

    def select_item(self):
        filters = self.get_filter()
        self.filtered_elements = filter(self.all_elements, filters)
        self.filtered_elements["datum"] = self.filtered_elements[
            "beginn_datum"
        ].dt.strftime("%d.%m.%Y")
        fields = [
            "datum",
            "status",
            "signatur",
            "titel",
            "urheber",
            "partei",
            "beginn_datum",
        ]
        settings = {
            "height": 400,
            "selection_mode": "single",
            "field": "pol_matters",
        }
        cols = [
            {
                "name": "beginn_datum",
                "hide": True,
                "width": 100,
                "precision": 0,
                "type": "date",
                format: "%d.%m.%Y",
            }
        ]
        st.subheader(
            f"{lang('pol_initiatives')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        st.markdown(lang("matters_table_help"))
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        cfg = {
            "filename": f"{lang('pol_matters')}.txt",
            "button_text": lang("download"),
        }
        show_download_button(self.filtered_elements, cfg)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("signatur").loc[
                sel_member["signatur"]
            ]
            pol_matter = PolMatter(self.parent, row, sel_member["signatur"])
            pol_matter.show_detail()

    def show_plot(self):
        def show_settings():
            cols = st.columns(2)
            with cols[0]:
                grouping_options = lang("figure1_grouping_options")
                self.settings_plot["groupby"] = st.selectbox(
                    label="Gruppierung", options=grouping_options
                )
                if grouping_options.index(self.settings_plot["groupby"]) == 0:
                    self.settings_plot["color"] = None
                elif grouping_options.index(self.settings_plot["groupby"]) == 1:
                    self.settings_plot["color"] = "thema_1"
                else:
                    self.settings_plot["color"] = "partei"

        def handle_groupby_none(df: pd.DataFrame):
            df.loc[:, "count"] = 1
            df_count = (
                df[["jahr_datum", "count"]].groupby(["jahr_datum"]).sum().reset_index()
            )
            self.settings_plot["y"] = "count"
            self.settings_plot["y_title"] = lang("figure1_title_none")
            self.settings_plot["fig_legend"] = lang("figure1_legend_groupby_none")
            return df_count

        def handle_groupby_theme(df: pd.DataFrame):
            df = df[["jahr_datum", "thema_1", "thema_2"]].copy()
            df["weight"] = df.apply(lambda row: 0.5 if row["thema_2"] else 1, axis=1)
            df_theme2 = df[["jahr_datum", "thema_2", "weight"]].copy().dropna()
            df_theme2.columns = ["jahr_datum", "thema_1", "weight"]
            df = pd.concat([df, df_theme2], ignore_index=True)
            # if a filte ris set on the theme, we need to filter again since in the original filter
            # all matters are found where 1 of the theme fields matches the theme.
            if len(self.settings_plot["themes"]) > 0:
                df = df[df["thema_1"].isin(self.settings_plot["themes"])]
            df = df.groupby(["jahr_datum", "thema_1"])["weight"].sum().reset_index()
            self.settings_plot["title"] = "Anzahl Vorstösse pro Jahr nach Thema"
            self.settings_plot["fig_legend"] = lang("figure1_legend_groupby_theme")
            self.settings_plot["y"] = "weight"

            return df

        def handle_groupby_party(df: pd.DataFrame):
            df.loc[:, "count"] = 1
            fields = ["jahr_datum", "partei", "count"]
            df = df[fields].groupby(["jahr_datum", "partei"]).sum().reset_index()
            self.settings_plot["y"] = "count"
            self.settings_plot["fig_legend"] = lang("figure1_legend_groupby_party")
            return df

        show_settings()
        filters = self.get_filter(excluded_list=["matter_title", "year", "status"])
        df = filter(self.all_elements, filters)
        if self.settings_plot["color"] is None:
            df = handle_groupby_none(df)
        elif self.settings_plot["color"] == "thema_1":
            df = handle_groupby_theme(df)
        elif self.settings_plot["color"] == "partei":
            df = handle_groupby_party(df)

        self.settings_plot["y_domain"] = [0, df[self.settings_plot["y"]].max()]
        # self.settings_plot["x_domain"] = [
        #    df["jahr_datum"].min(),
        #    df["jahr_datum"].max(),
        # ]

        self.settings_plot["tooltip"] = [
            self.settings_plot["x"],
            self.settings_plot["y"],
        ]
        if self.settings_plot["color"] is not None:
            self.settings_plot["tooltip"].append(self.settings_plot["color"])
        self.settings_plot[
            "title"
        ] = f"{lang('figure1_title_none')} {self.settings_plot['groupby']}"
        time_series_line(df, self.settings_plot)
        st.write(self.settings_plot["fig_legend"])


class Membership:
    def __init__(self):
        self.name = ""
        self.type = ""
        self.members = []
        self.pol_matters = []


class Memberships:
    def __init__(self, parent, df):
        self.parent = parent
        self.all_elements = self.get_elements(df)

    def filter(self, filters):
        for filter in filters:
            df = self.all_elements[
                self.all_elements[filter["field"]] == filter["value"]
            ]
        return df

    def get_elements(self, df):
        return df

    def get_filter(self, df, filters, parent):
        return df

    def select_item(self):
        filters = [
            "activ_member",
            "name",
            "party",
            "active_member",
            "electoral_district",
        ]
        self.filtered_elements = get_filter(self.all_elements, filters, self.parent)
        fields = [
            "kurzname_gre",
            "name_gre",
            "gremientyp",
            "gr_wahlkreis",
            "beginn_mit",
            "ende_mit",
            "funktion_adr",
            "uni_nr_gre",
            "uni_nr_adr",
        ]
        cols = []
        st.subheader(
            f"{lang('committee_members')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        st.markdown(lang("votations_table_help"))
        sel_member = show_table(self.filtered_elements[fields], cols, DEF_GRID_SETTING)
        cfg = {"filename": f"{lang('members')}.txt", "button_text": lang("download")}
        show_download_button(self.filtered_elements, cfg)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("uni_nr_gre").loc[
                sel_member["uni_nr_gre"]
            ]
            st.write("row:", row)
            # self.current_member = Member(self.parent, row)
            # self.current_member.show_detail()


class Parliament:
    def select_plot(self):
        plot_options = lang("figures_menu")
        selected_plot = st.sidebar.selectbox(
            label=lang("figures_label"), options=plot_options
        )
        if plot_options.index(selected_plot) == 0:
            self.votations.show_plot()
        elif plot_options.index(selected_plot) == 1:
            self.pol_matters.show_plot()
        elif plot_options.index(selected_plot) == 2:
            self.parties.show_plot()

    def __init__(self):
        with st.spinner(lang("loading_data")):
            placeholder = st.empty()
            placeholder.write(lang("loading_members"))
            df_all_members = get_table(Urls.MEMBERS_ALL.value)
            df_all_members["gr_beginn"] = pd.to_datetime(df_all_members["gr_beginn"])
            self.min_year = df_all_members["gr_beginn"].dt.year.min()
            self.max_year = datetime.now().year
            self.year_options = list(range(self.min_year, self.max_year + 1))
            self.members = Members(self, df_all_members)

            self.party_options = sorted(
                list(df_all_members["partei_kname"].dropna().unique())
            )
            self.electoral_district_options = list(
                df_all_members["gr_wahlkreis"].dropna().unique()
            )

            placeholder.write(lang("loading_matters"))
            df_matters = get_table(Urls.INITIATIVES.value)
            self.pol_matters = PolMatters(self, df_matters)
            self.pol_matters_themes = list(df_matters["thema_1"].unique()) + list(
                df_matters["thema_2"].unique()
            )
            self.matter_options = list(df_matters["geschaftstyp"].unique())

            placeholder.write(lang("loading_committees"))
            df_memberships = get_table(Urls.MEMBERSHIPS.value)
            self.memberships = Memberships(self, df_memberships)

            # holds urls from grosserrat.bs.ch for active committees
            df_url_bodies = pd.read_csv(DATA + "committee_url.csv", sep=";")
            df_bodies = get_table(Urls.BODIES.value)
            self.bodies = Bodies(self, df_memberships, df_url_bodies, df_bodies)
            self.body_type_options = list(df_memberships["gremientyp"].unique())
            df_bodies = (
                df_memberships[["kurzname_gre", "name_gre"]].drop_duplicates().dropna()
            )
            self.body_options = dict(
                zip(df_bodies["kurzname_gre"], df_bodies["name_gre"])
            )

            party_df = (
                df_all_members[["partei", "partei_kname"]].drop_duplicates().dropna()
            )
            self.parties_dict = dict(zip(party_df["partei_kname"], party_df["partei"]))
            df = pd.read_csv(DATA + "parties.tab", sep=";")
            self.parties = Parties(self, df)

            placeholder.write(lang("loading_documents"))
            df_docs = get_table(Urls.DOCUMENTS.value)
            df_summaries = pd.read_csv(DATA + "summaries.csv", sep="|")
            self.documents = Documents(self, df_docs, df_summaries)

            placeholder.write(lang("loading_votations"))
            # takes too much time, 135 MB as of 10/23, use votations.py to preprocess
            # self.votations = Votations(self, get_table(100186))
            df_matters = pd.read_parquet(DATA + "votation_matters.parquet")
            df_results = pd.read_parquet(DATA + "votation_results.parquet")
            self.votations = Votations(self, df_matters, df_results)
            placeholder.write("")
