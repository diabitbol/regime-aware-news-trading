#!/usr/bin/env python3
"""
Download and prepare daily WTI data for the HMM regime project.

Data sources
------------
FRED:
    DCOILWTICO  -> WTI spot price
    OVXCLS      -> CBOE Crude Oil ETF Volatility Index (OVX)

EIA:
    RCLC1       -> Cushing, OK crude-oil futures, Contract 1
    RCLC4       -> Cushing, OK crude-oil futures, Contract 4

Important
---------
EIA stopped publishing these NYMEX futures series after 2024-04-05.
Therefore the HMM feature file produced by this script ends around that date.

The term-structure feature is:

    TS_t = log(F1_t / F4_t)

Positive TS => backwardation
Negative TS => contango

Outputs
-------
data/wti_hmm_raw.csv
data/wti_hmm_features.csv
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests


START_DATE = "2007-05-10"
END_DATE = "2026-10-14"

OUT_DIR = Path("data")
RAW_FILE = OUT_DIR / "wti_hmm_raw.csv"
FEATURE_FILE = OUT_DIR / "wti_hmm_features.csv"

FRED_SERIES = {
    "WTI_SPOT": "DCOILWTICO",
    "OVX": "OVXCLS",
}

EIA_FUTURES = {
    "F1": "https://www.eia.gov/dnav/pet/hist_xls/RCLC1d.xls",
    "F4": "https://www.eia.gov/dnav/pet/hist_xls/RCLC4d.xls",
}


def download_fred(series_id: str, name: str) -> pd.DataFrame:
    """Download one FRED series as CSV, no API key required."""

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={START_DATE}&coed={END_DATE}"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))

    if len(df.columns) != 2:
        raise ValueError(
            f"Unexpected FRED response for {series_id}: "
            f"{df.columns.tolist()}"
        )

    df.columns = ["date", name]

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df[name] = pd.to_numeric(
        df[name],
        errors="coerce"
    )

    return (
        df.dropna(subset=["date"])
        .set_index("date")
        .sort_index()
    )


def _find_eia_data_sheet(
    excel_bytes: bytes,
    name: str
) -> pd.DataFrame:
    """
    Parse an EIA historical XLS robustly.

    EIA workbooks normally contain a sheet called 'Data 1',
    with title/source rows above the actual Date/value table.
    """

    workbook = pd.ExcelFile(
        io.BytesIO(excel_bytes),
        engine="xlrd"
    )

    for sheet in workbook.sheet_names:

        raw = pd.read_excel(
            io.BytesIO(excel_bytes),
            sheet_name=sheet,
            header=None,
            engine="xlrd",
        )

        for i in range(min(len(raw), 30)):

            row = (
                raw.iloc[i]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            date_cols = [
                j
                for j, value in enumerate(row)
                if value in {"date", "day"}
            ]

            if not date_cols:
                continue

            date_col = date_cols[0]

            candidate_cols = [
                j
                for j in range(date_col + 1, raw.shape[1])
                if not raw.iloc[i + 1 :, j].dropna().empty
            ]

            if not candidate_cols:
                continue

            value_col = candidate_cols[0]

            df = raw.iloc[
                i + 1 :,
                [date_col, value_col]
            ].copy()

            df.columns = ["date", name]

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            df[name] = pd.to_numeric(
                df[name],
                errors="coerce"
            )

            df = df.dropna(
                subset=["date", name]
            )

            if len(df) > 100:
                return (
                    df.set_index("date")
                    .sort_index()
                )

    raise ValueError(
        f"Could not find a usable Date/value table "
        f"in EIA workbook for {name}."
    )


def download_eia_future(
    url: str,
    name: str
) -> pd.DataFrame:
    """Download one official EIA NYMEX futures history spreadsheet."""

    r = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    r.raise_for_status()

    df = _find_eia_data_sheet(
        r.content,
        name
    )

    return df.loc[START_DATE:END_DATE]


def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # DOWNLOAD RAW SERIES
    # ---------------------------------------------------------

    print("Downloading WTI spot from FRED...")

    wti = download_fred(
        FRED_SERIES["WTI_SPOT"],
        "WTI_SPOT"
    )

    print("Downloading OVX from FRED...")

    ovx = download_fred(
        FRED_SERIES["OVX"],
        "OVX"
    )

    print("Downloading WTI futures Contract 1 from EIA...")

    f1 = download_eia_future(
        EIA_FUTURES["F1"],
        "F1"
    )

    print("Downloading WTI futures Contract 4 from EIA...")

    f4 = download_eia_future(
        EIA_FUTURES["F4"],
        "F4"
    )

    # ---------------------------------------------------------
    # SAVE RAW DATA
    # ---------------------------------------------------------
    #
    # Here an outer join is useful:
    # we want to preserve every date from every source
    # for inspection/debugging.
    #

    raw = pd.concat(
        [wti, ovx, f1, f4],
        axis=1,
        join="outer"
    ).sort_index()

    raw = raw.loc[
        START_DATE:END_DATE
    ]

    raw.to_csv(
        RAW_FILE
    )

    # =========================================================
    # FEATURE CONSTRUCTION
    # =========================================================

    # ---------------------------------------------------------
    # 1. WTI RETURNS + REALIZED VOLATILITY
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # These features are calculated BEFORE merging WTI with
    # the other datasets.
    #
    # Therefore shift(1) means:
    # previous available WTI observation,
    #
    # and rolling(20) means:
    # previous 20 WTI observations.
    #

    wti_features = (
        wti[["WTI_SPOT"]]
        .dropna()
        .copy()
    )

    # Logarithms require strictly positive prices.
    wti_features = wti_features[
        wti_features["WTI_SPOT"] > 0
    ].copy()

    # Daily log return
    wti_features["r_t"] = np.log(
        wti_features["WTI_SPOT"]
        /
        wti_features["WTI_SPOT"].shift(1)
    )

    # 20-observation realized volatility,
    # annualized with sqrt(252)
    wti_features["sigma20"] = (
        wti_features["r_t"]
        .rolling(
            window=20,
            min_periods=20
        )
        .std()
        * np.sqrt(252)
    )

    wti_features = (
        wti_features[
            ["r_t", "sigma20"]
        ]
        .dropna()
    )

    # ---------------------------------------------------------
    # 2. FUTURES TERM STRUCTURE
    # ---------------------------------------------------------
    #
    # F1 and F4 are first aligned with EACH OTHER only.
    #

    futures = pd.concat(
        [f1, f4],
        axis=1,
        join="inner"
    ).dropna()

    futures = futures[
        (futures["F1"] > 0)
        &
        (futures["F4"] > 0)
    ].copy()

    futures["TS"] = np.log(
        futures["F1"]
        /
        futures["F4"]
    )

    ts_features = futures[
        ["TS"]
    ].copy()

    # ---------------------------------------------------------
    # 3. OVX
    # ---------------------------------------------------------

    ovx_features = (
        ovx[["OVX"]]
        .dropna()
        .copy()
    )

    # ---------------------------------------------------------
    # 4. FINAL ALIGNMENT
    # ---------------------------------------------------------
    #
    # Only NOW do we intersect the calendars.
    #
    # This prevents missing dates in F1/F4/OVX from destroying
    # the WTI rolling-volatility computation.
    #

    features = pd.concat(
        [
            wti_features,
            ts_features,
            ovx_features
        ],
        axis=1,
        join="inner"
    )

    features = (
        features
        .dropna()
        .sort_index()
    )

    features.to_csv(
        FEATURE_FILE
    )

    # =========================================================
    # DIAGNOSTICS
    # =========================================================

    print("\nDone.")

    print(
        f"Raw data:     "
        f"{RAW_FILE.resolve()}"
    )

    print(
        f"HMM features: "
        f"{FEATURE_FILE.resolve()}"
    )

    print("\nRaw observations:")

    print(
        raw.notna().sum()
    )

    print("\nIntermediate feature counts:")

    print(
        f"WTI return/vol observations: "
        f"{len(wti_features):,}"
    )

    print(
        f"Term-structure observations: "
        f"{len(ts_features):,}"
    )

    print(
        f"OVX observations: "
        f"{len(ovx_features):,}"
    )

    if len(features):

        print(
            "\nUsable feature period: "
            f"{features.index.min().date()} "
            f"-> "
            f"{features.index.max().date()}"
        )

        print(
            "Number of usable daily observations: "
            f"{len(features):,}"
        )

    else:

        print(
            "WARNING: no complete feature rows "
            "were produced."
        )

    print(
        "\nFeature summary:"
    )

    print(
        features.describe().T
    )

    print(
        "\nNOTE: EIA stopped publishing its NYMEX "
        "futures history after 2024-04-05, "
        "so TS and therefore the complete HMM "
        "dataset end around that date."
    )


if __name__ == "__main__":
    main()