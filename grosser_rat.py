from typing import Any
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
from lang import get_lang
from helper import show_table, randomword
from enum import Enum, auto


PAGE = __name__
MEMBER_COUNT = 100
DEF_GRID_SETTING = {
    "fit_columns_on_grid_load": True,
    "height": 400,
    "selection_mode": "single",
}


class Urls(Enum):
    MEMBERS_ALL = 100307
    INITIATIVES = 100086
    VOTATIONS = 100186
    MEMBERSHIPS = 100308
    DOCUMENTS = 100313


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
        if filter["operator"] == "isin" and type(filter['field'] == list):
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
            df = df[df[filter["field"]].str.contains(filter["value"], case=False)]
        elif filter["operator"] == "isin":
            df = df[df[filter["field"]].isin(filter["value"], case=False)]
    return df


def get_table(table_nr: int):
    URL = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/{table_nr}/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B"
    df = pd.read_csv(URL, sep=";")
    return df


class Document:
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
        self.summary = row["summary"]

    def show_detail(self):
        st.markdown(f"{lang('document')}: **{self.title}**")
        tabs = st.tabs(lang("document_tabs"))
        with tabs[0]:
            fields = [
                "dokudatum",
                "dok_laufnr",
                "titel_dok",
                "laufnr_ges",
                "signatur_ges",
                "titel_ges",
                "url_ges",
                "url_dok",
            ]
            df = self.data[fields].copy()
            df["status"] = [
                lang("in_progress") if x == "A" else lang("closed")
                for x in df["status"]
            ]
            df.columns = [
                "dokudatum",
                "dok_laufnr",
                "titel_dok",
                "laufnr_ges",
                "signatur_ges",
                "titel_ges",
                "url_ges",
            ]
            df = pd.DataFrame(df.T).reset_index().dropna()
            df.columns = lang("field_value_cols")
            table = df.to_html(index=False)
            # st.markdown(table, unsafe_allow_html=True)
        with tabs[1]:
            cols = st.columns([2, 10])
            with cols[0]:
                ...
            with cols[1]:
                ...
            with cols[0]:
                ...
            with cols[1]:
                ...


class Documents:
    def __init__(self, parent, df):
        self.parent = parent
        self.all_elements = self.get_elements(df)

    def get_elements(self, df):
        fields = [
            "dokudatum",
            "dok_laufnr",
            "titel_dok",
            "laufnr_ges",
            "signatur_ges",
            "titel_ges",
            "url_ges",
            "url_dok",
        ]
        df = df[fields]
        df["dokudatum"] = pd.to_datetime(df["dokudatum"])
        df["year"] = df["dokudatum"].dt.year
        return df[fields]

    def select_item(self):
        filters = ["year"]
        self.filtered_elements = get_filter(self.all_elements, filters, self.parent)
        fields = [
            "dokudatum",
            "titel_dok",
            "signatur_ges",
            "titel_ges",
        ]
        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "field": "documents",
        }
        cols = []
        st.subheader(
            f"{lang('documents')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("dok_laufnr").loc[
                sel_member["dok_laufnr"]
            ]
            doc = Document(self.parent, row, sel_member["dok_laufnr"])
            doc.show_detail()

    def filter(self, filters):
        for filter in filters:
            df = self.all_elements[
                self.all_elements[filter["field"]] == filter["value"]
            ]
        return df


class Body:
    def __init__(self, parent, row, id):
        self.parent = parent
        self.data = row
        self.name = row["name_gre"]
        self.short_name = row["kurzname_gre"]
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
            settings = {
                "fit_columns_on_grid_load": False,
                "height": 400,
                "selection_mode": "single",
                "field": randomword(10),
            }
            st.markdown(
                f"**{lang('committee_members')} {self.name}: {len(self.members)}**"
            )
            show_table(self.members, cols, settings)


class Bodies:
    def __init__(self, parent, df_bodies, df_urls):
        self.parent = parent
        self.memberships = df_bodies
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
        return filters

    def get_elements(self, df):
        fields = ["uni_nr_gre", "kurzname_gre", "name_gre", "gremientyp"]
        # add and remove fields here
        df = df[fields].drop_duplicates()
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
        return df

    def select_item(self):
        filters = self.get_filter()
        self.filtered_elements = filter(self.all_elements, filters)
        fields = ["kurzname_gre", "name_gre", "gremientyp", "uni_nr_gre"]
        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "field": "bodies",
        }
        cols = []
        st.subheader(
            f"{lang('parliament_committees')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("uni_nr_gre").loc[
                sel_member["uni_nr_gre"]
            ]
            self.current_body = Body(self.parent, row, sel_member["uni_nr_gre"])
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

        self.all_elements = self.get_elements(df_matters)
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
            # Geschäft
            matter = st.text_input(lang("matter"))
            if matter > "":
                filters = add_filter(filters, "Geschaeft", matter, "contains")

            # election year
            year_options = self.parent.year_options
            year_options = [lang("all")] + list(year_options)
            year = st.selectbox(lang("year"), year_options)
            if year_options.index(year) > 0:
                filters = add_filter(filters, "jahr", year)

            # election year
            result_options = [lang("all"), lang("accepted"), lang("rejected")]
            result = st.selectbox(lang("result"), result_options)
            if result_options.index(result) > 0:
                filters = add_filter(filters, "Angenommen", result)
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
        df["pzt_ja"] = df["Ja-Stimmen"] / (MEMBER_COUNT - df["Abwesende"]) * 100.0
        df["pzt_abwesend"] = df["Abwesende"] / MEMBER_COUNT * 100
        df["Angenommen"] = df["Ja-Stimmen"] > df["Nein-Stimmen"]
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
            "Datum",
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
        st.subheader(f"{lang('votations')} ({len(df)}/{len(self.all_elements)})")
        sel_member = show_table(df, cols, settings)
        if len(sel_member) > 0:
            row = self.filtered_elements[
                self.filtered_elements["id"] == sel_member["id"]
            ].iloc[0]
            self.current_body = Votation(self.parent, row, sel_member["id"])
            self.current_body.show_detail()


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
        self.pol_matters = self.parent.pol_matters.filter(
            [{"field": "urheber", "value": f"{self.name}, {self.first_name}"}]
        )
        self.memberships = self.parent.memberships.filter(
            [{"field": "uni_nr_adr", "value": self.uni_nr}]
        )
        self.votations = self.parent.votations.filter_results(
            [
                {"field": "Name", "value": self.full_name},
                {"field": "Sitz Nr.", "value": self.place_id},
            ],
            add_details=True,
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
        df["gr_beginn"] = pd.to_datetime(df["gr_beginn"])
        df["gr_ende"] = pd.to_datetime(df["gr_ende"])
        df["wahljahr"] = df["gr_beginn"].dt.year.astype(int)
        df["austritt"] = df["gr_ende"].dt.year.fillna(0).astype(int)
        df["gr_sitzplatz"] = df["gr_sitzplatz"].fillna(0).astype(int)
        df["geschlecht"] = ["M" if x == "Herr" else "F" for x in df["anrede"]]
        df["active_member"] = [
            True if x == "Ja" else False for x in df["ist_aktuell_grossrat"]
        ]
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
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "field": "members",
        }
        cols = []
        st.subheader(
            f"{lang('parliament_members')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        df = self.filtered_elements[fields].copy()
        sel_member = show_table(df, cols, settings)
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
        self.summary = row["summary"]
        self.documents = self.parent.documents.filter(
            [{"field": "signatur_ges", "value": signatur}]
        ).sort_values(by="dokudatum")

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
            df["status"] = [
                lang("in_progress") if x == "A" else lang("closed")
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
            st.link_button(
                lang("matter"), self.url_matter, help=lang("more_info_pol_matter")
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
        self.theme_options = self.get_themes(self.all_elements)

    def get_themes(self, df: pd.DataFrame)->list:
        concat_list = list(df["thema_1"].unique()) + list(df["thema_2"].unique())
        concat_list = [x for x in concat_list if x is not np.nan]
        return sorted(concat_list)
        
    def get_filter(self):
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
            name = st.text_input(lang("matter"))
            if name > "":
                filters = add_filter(filters, "titel", name, "contains")

            # year
            year_options = [lang("year")] + self.parent.year_options
            year = st.selectbox(lang("committee_type"), options=year_options)
            if year_options.index(year) > 0:
                filters = add_filter(filters, "jahr", year)

            # political party
            party_options = [lang("all")] + self.parent.party_options
            party = st.selectbox(lang("party"), options=party_options)
            if party_options.index(party) > 0:
                filters = add_filter(filters, "partei", party)

            # theme
            themes = st.multiselect(
                lang("theme"),
                self.theme_options,
                help="Selektiere ein oder mehrere Themen",
            )
            if len(themes) > 0:
                filters = add_filter(filters, ["thema_1", "thema_1"], themes, "isin")

            # Status
            status_options = [lang("all"), lang("in_progress"), lang("closed")]
            status = st.selectbox(lang("status"), options=status_options)
            if status_options.index(status) == 1:
                filters = add_filter(filters, "status", "B")
            elif status_options.index(status) == 2:
                filters = add_filter(filters, "status", "A")

        return filters

    def get_elements(self, df):
        df_urls = pd.read_csv("./matter_url.csv", sep=";")
        df = df.merge(df_urls, on="signatur", how="left")
        df["beginn_datum"] = pd.to_datetime(df["beginn_datum"])
        df["jahr"] = df["beginn_datum"].dt.year
        return df

    def select_item(self):
        filters = self.get_filter()
        self.filtered_elements = filter(self.all_elements, filters)
        fields = [
            "beginn_datum",
            "status",
            "signatur",
            "titel",
            "urheber",
            "partei",
        ]
        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "field": "pol_matters",
        }
        cols = []
        st.subheader(
            f"{lang('pol_initiatives')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("signatur").loc[
                sel_member["signatur"]
            ]
            pol_matter = PolMatter(self.parent, row, sel_member["signatur"])
            pol_matter.show_detail()

    def filter(self, filters):
        for filter in filters:
            df = self.all_elements[
                self.all_elements[filter["field"]] == filter["value"]
            ]
        return df


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
        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "field": randomword(10),
        }
        cols = []
        st.subheader(
            f"{lang('committee_members')} ({len(self.filtered_elements)}/{len(self.all_elements)})"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("uni_nr_gre").loc[
                sel_member["uni_nr_gre"]
            ]
            st.write("row:", row)
            # self.current_member = Member(self.parent, row)
            # self.current_member.show_detail()


class Parliament:
    def __init__(self):
        with st.spinner(lang("loading_data")):
            text = st.empty()
            text.write(lang("loading_members"))
            df_all_members = get_table(Urls.MEMBERS_ALL.value)
            df_all_members["gr_beginn"] = pd.to_datetime(df_all_members["gr_beginn"])
            self.min_year = df_all_members["gr_beginn"].dt.year.min()
            self.max_year = datetime.now().year
            self.year_options = list(range(self.min_year, self.max_year + 1))
            self.members = Members(self, df_all_members)

            self.party_options = list(df_all_members["partei_kname"].unique())
            self.electoral_district_options = list(
                df_all_members["gr_wahlkreis"].unique()
            )

            text.write(lang("loading_matters"))
            df_matters = get_table(Urls.INITIATIVES.value)
            self.pol_matters = PolMatters(self, df_matters)
            self.pol_matters_themes = list(df_matters["thema_1"].unique()) + list(
                df_matters["thema_2"].unique()
            )

            text.write(lang("loading_committees"))
            df_memberships = get_table(Urls.MEMBERSHIPS.value)
            self.memberships = Memberships(self, df_memberships)

            # holds urls from grosserrat.bs.ch for active committees
            df_url_bodies = pd.read_csv("./committee_url.csv", sep=";")
            self.bodies = Bodies(self, df_memberships, df_url_bodies)
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
            self.parties = dict(zip(party_df["partei_kname"], party_df["partei"]))

            text.write(lang("loading_documents"))
            self.documents = Documents(self, get_table(Urls.DOCUMENTS.value))

            text.write(lang("loading_votations"))
            # takes too much time, 135 MB as of 10/23, use votations.py to preprocess
            # self.votations = Votations(self, get_table(100186))
            df_matters = pd.read_parquet("./votation_matters.parquet")
            df_results = pd.read_parquet("./votation_results.parquet")
            self.votations = Votations(self, df_matters, df_results)
