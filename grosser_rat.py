import streamlit as st
from datetime import datetime
import pandas as pd
from lang import get_lang
from helper import show_table, randomword 

PAGE = __name__


def lang(text):
    return get_lang(PAGE)[text]


def get_filter(df: pd.DataFrame, filters: list, parliament):
    with st.sidebar.expander(f"🔎{lang('filter')}", expanded=True):
        if "member_name" in filters:
            name = st.text_input(lang('member_name'))
            df = df[
                df["name"].str.contains(name, case=False)
                | df["vorname"].str.contains(name, case=False)
            ]
        if "election_year" in filters:
            year_options = parliament.year_options
            year_options = [lang('all')] + list(year_options)
            year = st.selectbox("Wahljahr", year_options)
            if year_options.index(year) > 0:
                df = df[df["wahljahr"] == year]

        if "active_member" in filters:
            member_options = [lang('all'), "Aktive", "Ehemalige"]
            is_member = st.selectbox(lang('membership_status'), options=member_options)
            if member_options.index(is_member) == 1:
                df = df[df["active_member"] is True]
            elif member_options.index(is_member) == 2:
                df = df[df["active_member"] is False]

        if "electoral_district" in filters:
            district_options = [lang('all')] + parliament.electoral_district_options
            district = st.selectbox(lang('voting_district'), district_options)
            if district_options.index(district) > 0:
                df = df[df["gr_wahlkreis"] == district]

        if "committee_type" in filters:
            committee_type_options = [lang('all')] + parliament.committee_type_options
            body_type = st.selectbox(lang('committee_type'), options=committee_type_options)
            if committee_type_options.index(body_type) > 0:
                df = df[df["gremientyp"] == body_type]
        if "theme" in filters:
            theme_options = [lang('all')] + parliament.pol_matters_themes
            theme = st.selectbox(lang('theme'), options=theme_options)
            if theme_options.index(theme) > 0:
                df = df[(df["thema_1"] == theme) | (df["thema_2"] == theme)]
        if "member_party" in filters:
            party_options = [lang('all')] + parliament.party_options
            party = st.selectbox(lang('party'), options=party_options)
            if party_options.index(party) > 0:
                df = df[df["partei_kname"] == party]
    return df


def get_table(table_nr: int):
    URL = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/{table_nr}/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B"
    df = pd.read_csv(URL, sep=";")
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
            st.dataframe(series)
        with tabs[1]:
            cols = []
            settings = {
                "fit_columns_on_grid_load": False,
                "height": 400,
                "selection_mode": "single",
                "key": randomword(10),
            }
            st.markdown(f"**{lang('committee_members')} {self.name}: {len(self.members)}**")
            show_table(self.members, cols, settings)


class Bodies:
    def __init__(self, parent, df_bodies):
        self.parent = parent
        self.memberships = df_bodies
        self.all_elements = self.get_elements(df_bodies)
        self.filtered_elements = pd.DataFrame()

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
        filters = [
            {"body_name": self.parent.body_options},
            {"committee_type": self.parent.bodytype_options},
        ]
        self.filtered_elements = get_filter(self.all_elements, filters, self.parent)
        fields = ["kurzname_gre", "name_gre", "gremientyp", "uni_nr_gre"]
        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "key": "bodies",
        }
        cols = []
        st.subheader(
            f"{lang('parliament_committees')} ({len(self.filtered_elements)}/{len(self.all_elements)}"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        if len(sel_member) > 0:
            row = self.filtered_elements.set_index("uni_nr_gre").loc[
                sel_member["uni_nr_gre"]
            ]
            self.current_body = Body(self.parent, row, sel_member["uni_nr_gre"])
            self.current_body.show_detail()


class Member:
    def __init__(self, parent, row, id):
        self.parent = parent
        if type(row) == pd.DataFrame:
            row = row.sort_values(by="gr_beginn", ascending=False).iloc[0]
        self.data = pd.DataFrame(row).T.reset_index()
        self.name = row["name"]
        self.first_name = row["vorname"]
        self.party = row["partei_kname"]
        self.uni_nr = id
        self.pol_matters = self.parent.pol_matters.filter(
            [{"key": "urheber", "value": f"{self.name}, {self.first_name}"}]
        )
        self.memberships = self.parent.memberships.filter(
            [{"key": "uni_nr_adr", "value": self.uni_nr}]
        )

    def show_detail(self):
        st.markdown(f"**Mitglied: {self.name} {self.first_name}**")
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
            df = self.data[fields].copy().T.reset_index()
            df.columns = lang('field_value_cols')
            df.dropna(inplace=True)
            table = df.to_html(index=False)
            st.markdown(table, unsafe_allow_html=True)
        with tabs[1]:
            fields = ["signatur", "geschaftstyp", "titel", "beginn_datum", "ende"]
            cols = []
            settings = {
                "fit_columns_on_grid_load": False,
                "height": 400,
                "selection_mode": "single",
                "key": randomword(10),
            }
            self_matter = show_table(self.pol_matters[fields], cols, settings)
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
            settings = {
                "fit_columns_on_grid_load": False,
                "height": 400,
                "selection_mode": "single",
                "key": randomword(10),
            }
            sel_membership = show_table(self.memberships[fields], cols, settings)
            if len(sel_membership) > 0:
                row = self.memberships[
                    self.memberships["uni_nr_gre"] == sel_membership["uni_nr_gre"]
                ].iloc[0]
                # matter = PolMatter(self.parent, row)
                # matter.show_detail()


class Members:
    def __init__(self, parent, df_members):
        self.parent = parent
        self.all_elements = self.get_elements(df_members)
        self.filtered_elements = pd.DataFrame()

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
        df["wahljahr"] = df["gr_beginn"].dt.year
        df["austritt"] = df["gr_ende"].dt.year
        df["geschlecht"] = ["m" if x == "Herr" else "f" for x in df["anrede"]]
        df["active_member"] = [
            True if x == "Ja" else False for x in df["ist_aktuell_grossrat"]
        ]
        return df

    def select_item(self):
        filters = [
            "activ_member",
            "member_name",
            "member_party",
            "active_member",
            "election_year",
            "electoral_district",
        ]
        self.filtered_elements = get_filter(self.all_elements, filters, self.parent)
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
            "key": "members",
        }
        cols = []
        st.subheader(
            f"{lang('parliament_members')} ({len(self.filtered_elements)}/{len(self.all_elements)}"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
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

    def show_detail(self):
        st.markdown(f"{lang('matter')}: **{self.title}**")
        tabs = st.tabs(lang('matter_tabs'))
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
                lang('in_progress') if x == "A" else lang('closed') for x in df["status"]
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
            df.columns = lang('field_value_cols')
            table = df.to_html(index=False)
            st.markdown(table, unsafe_allow_html=True)
        with tabs[1]:
            cols = st.columns([2, 10])
            with cols[0]:
                st.link_button(lang('matter'), self.url_matter)
            with cols[1]:
                st.markdown(lang('all_info_docs_matter'))
            with cols[0]:
                st.link_button(lang('document'), self.url_doc)
            with cols[1]:
                st.markdown(lang('direct_link_pdf'))
        #with tabs[2]:
        #    st.markdown(lang('text'])
        #with tabs[3]:
        #    st.markdown(lang('word_cloud'])


class PolMatters:
    def __init__(self, parent, df):
        self.parent = parent
        self.all_elements = self.get_elements(df)

    def get_elements(self, df):
        df_urls = pd.read_csv("./matter_url.csv", sep=";")
        df = df.merge(df_urls, on="signatur", how="left")
        # df_summary = pd.read_csv("./summaries.csv", sep="|")
        # df = df.merge(df_summary, on="signatur", how="left")
        return df

    def select_item(self):
        filters = [
            "matter_title",
            'year',
            "author",
            "theme"
        ]
        self.filtered_elements = get_filter(self.all_elements, filters, self.parent)
        fields = ["signatur", "titel", "partei", "urheber", "beginn_datum", "ende"]
        settings = {
            "fit_columns_on_grid_load": False,
            "height": 400,
            "selection_mode": "single",
            "key": "pol_matters",
        }
        cols = []
        st.subheader(
            f"{lang('pol_initiatives')} ({len(self.filtered_elements)}/{len(self.all_elements)}"
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
            df = self.all_elements[self.all_elements[filter["key"]] == filter["value"]]
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
            df = self.all_elements[self.all_elements[filter["key"]] == filter["value"]]
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
            "key": randomword(10),
        }
        cols = []
        st.subheader(
            f"{lang('committee_members')} ({len(self.filtered_elements)}/{len(self.all_elements)}"
        )
        sel_member = show_table(self.filtered_elements[fields], cols, settings)
        if len(sel_member) > 0:
            st.write(self.filtered_elements)
            row = self.filtered_elements.set_index("uni_nr_gre").loc[
                sel_member["uni_nr_gre"]
            ]
            st.write("row:", row)
            # self.current_member = Member(self.parent, row)
            # self.current_member.show_detail()


class Parliament:
    def __init__(self):
        df_all_members = get_table(100307)
        df_all_members["gr_beginn"] = pd.to_datetime(df_all_members["gr_beginn"])
        self.min_year = df_all_members["gr_beginn"].dt.year.min()
        self.max_year = datetime.now().year
        self.year_options = list(range(self.min_year, self.max_year + 1))
        self.members = Members(self, df_all_members)

        self.party_options = list(df_all_members["partei_kname"].unique())
        self.electoral_district_options = list(df_all_members["gr_wahlkreis"].unique())

        df_matters = get_table(100086)
        self.pol_matters = PolMatters(self, df_matters)
        self.pol_matters_themes = list(df_matters["thema_1"].unique()) + list(
            df_matters["thema_2"].unique()
        )

        df_bodies = get_table(100308)
        self.bodies = Bodies(self, df_bodies)
        self.bodytype_options = list(df_bodies["gremientyp"].unique())
        df_bodies = df_bodies[["kurzname_gre", "name_gre"]].drop_duplicates().dropna()
        self.body_options = dict(zip(df_bodies["kurzname_gre"], df_bodies["name_gre"]))

        party_df = df_all_members[["partei", "partei_kname"]].drop_duplicates().dropna()
        self.parties = dict(zip(party_df["partei_kname"], party_df["partei"]))

        df_memberships = get_table(100308)
        self.memberships = Memberships(self, df_memberships)
