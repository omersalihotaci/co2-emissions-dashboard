"""Data loading and preprocessing for the CO2 dashboard.

Source: Our World in Data - CO2 and Greenhouse Gas Emissions dataset
https://github.com/owid/co2-data
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_FILE = DATA_DIR / "owid-co2-data.csv"
CONTINENT_FILE = DATA_DIR / "continents.csv"

START_YEAR = 1990

COLUMNS = [
    "country", "year", "iso_code", "population", "gdp",
    "co2", "co2_per_capita", "co2_per_gdp", "share_global_co2",
    "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2",
    "energy_per_capita", "primary_energy_consumption",
    "methane_per_capita", "nitrous_oxide_per_capita",
]
FUELS = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"]
FUEL_LABELS = {
    "coal_co2": "Coal", "oil_co2": "Oil", "gas_co2": "Gas",
    "cement_co2": "Cement", "flaring_co2": "Flaring", "other_co2": "Other / unattributed",
}


def missing_report(df: pd.DataFrame, cols) -> pd.Series:
    return (df[cols].isna().mean() * 100).round(1)


def load_and_clean():
    """Return (clean_df, report) where report documents every cleaning step."""
    raw = pd.read_csv(RAW_FILE, usecols=COLUMNS)
    report = {"raw_shape": raw.shape}

    # 1) Keep real countries only: aggregates (World, Asia, income groups,
    #    "International shipping", ...) have no ISO code.
    df = raw[raw["iso_code"].notna()].copy()
    report["aggregates_removed"] = int(raw["iso_code"].isna().sum())

    # 2) Restrict to the modern period, where coverage is good.
    df = df[df["year"] >= START_YEAR]
    report["after_filter_shape"] = df.shape

    # 3) Attach continents (OWID definition). Antarctica has no continent row.
    cont = pd.read_csv(CONTINENT_FILE)[["code", "owid_region"]]
    cont = cont.rename(columns={"code": "iso_code", "owid_region": "continent"})
    df = df.merge(cont, on="iso_code", how="left")
    report["no_continent"] = sorted(df.loc[df["continent"].isna(), "country"].unique())
    df = df[df["continent"].notna()]

    value_cols = [c for c in COLUMNS if c not in ("country", "year", "iso_code")]
    report["missing_before"] = missing_report(df, value_cols)

    # 4) Fill gaps *inside* each country's time series by linear interpolation.
    #    We never extrapolate beyond the first/last observed year.
    df = df.sort_values(["country", "year"])
    interp_cols = ["population", "gdp", "energy_per_capita", "primary_energy_consumption"]
    df[interp_cols] = (
        df.groupby("country")[interp_cols]
        .transform(lambda s: s.interpolate(limit_area="inside"))
    )

    # 5) Fuel breakdown is not reported for many small countries. Missing fuel
    #    values are set to 0 and the remainder of total CO2 is kept as
    #    "other_co2" so that totals always add up.
    df[FUELS] = df[FUELS].fillna(0)
    df["other_co2"] = (df["co2"] - df[FUELS].sum(axis=1)).clip(lower=0)

    # 6) Rows with no CO2 at all carry no information for this dashboard.
    report["no_co2_rows"] = int(df["co2"].isna().sum())
    df = df[df["co2"].notna()]

    # 7) Derived features.
    df["gdp_per_capita"] = df["gdp"] / df["population"]
    for f in FUELS + ["other_co2"]:
        df[f + "_share"] = np.where(df["co2"] > 0, df[f] / df["co2"] * 100, 0)

    report["missing_after"] = missing_report(df, value_cols + ["gdp_per_capita"])

    # 8) Outlier detection (IQR rule) on CO2 per capita. Outliers are real
    #    (petro-states, tiny economies), so they are flagged, not removed.
    q1, q3 = df["co2_per_capita"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    df["is_outlier"] = df["co2_per_capita"] > upper
    report["iqr_upper"] = float(upper)
    report["outlier_rows"] = int(df["is_outlier"].sum())
    report["outlier_countries"] = (
        df[df["is_outlier"]].groupby("country").size().sort_values(ascending=False)
    )
    report["clean_shape"] = df.shape
    return df.reset_index(drop=True), report
