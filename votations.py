import pandas as pd


def get_data():
    votations = pd.read_csv("./data/100186.csv", sep=";")

    votations["Datum"] = pd.to_datetime(votations["Datum"])
    votations["Jahr"] = votations["Datum"].dt.year
    votations["Traktandum"] = votations["Traktandum"].fillna(0)
    votations["Subtraktandum"] = votations["Subtraktandum"].fillna(0)
    # reformat index as year month abstnr, trakt, subtract: 20220147000600
    votations["id"] = (
        votations["Jahr"] * 1e10
        + votations["Datum"].dt.month * 1e8
        + votations["Abstimmungsnummer"] * 1e5
        + votations["Traktandum"] * 1e2
        + votations["Subtraktandum"]
    )
    matter_fields = [
        "Abstimmungsnummer",
        "Jahr",
        "Datum",
        "Zeit",
        "Traktandum",
        "Subtraktandum",
        "Geschaeft",
        "Typ",
        "Ja-Stimmen",
        "Nein-Stimmen",
        "Enthaltungen",
        "Abwesende",
        "Präsidiumsstimmen",
        "id",
    ]
    df_votation_matters = votations[matter_fields].drop_duplicates()
    df_votation_matters["Enthaltungen"] = df_votation_matters["Enthaltungen"].fillna(0)
    df_votation_matters["Abwesende"] = df_votation_matters["Abwesende"].fillna(0)
    df_votation_matters["Ja-Stimmen"] = df_votation_matters["Ja-Stimmen"].fillna(0)
    df_votation_matters["Nein-Stimmen"] = df_votation_matters["Nein-Stimmen"].fillna(0)
    df_votation_matters["Enthaltungen"] = df_votation_matters["Enthaltungen"].fillna(0)
    df_votation_matters["Präsidiumsstimmen"] = df_votation_matters[
        "Präsidiumsstimmen"
    ].fillna(0)
    df_votation_matters.to_parquet("./votation_matters.parquet", index=False)
    votations.rename(columns={"id": "id_geschaeft"}, inplace=True)
    votations_fields = [
        "Datum",
        "Jahr",
        "Sitz Nr.",
        "Name",
        "Fraktion",
        "Entscheid Mitglied",
        "Zeitstempel",
        "id_geschaeft",
    ]

    votations[votations_fields].to_parquet("./votation_results.parquet", index=False)


get_data()
