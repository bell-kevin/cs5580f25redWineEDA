"""Utilities for exploring the red wine quality dataset.

This module loads the dataset used in the accompanying exploratory data
analysis report and prints summary statistics that are incorporated into the
LaTeX document.  The script is intentionally read-only so that it can be run in
restricted execution environments that disallow writing to disk.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "red_wine_quality.csv"


def load_dataset() -> pd.DataFrame:
    """Load the red wine quality dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the red wine quality data with snake_case column
        names for easier downstream processing.
    """
    df = pd.read_csv(DATA_PATH)
    df.columns = [
        column.strip().lower().replace(" ", "_")
        for column in df.columns
    ]
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print key summary statistics for the dataset.

    Parameters
    ----------
    df:
        DataFrame containing the red wine quality observations.
    """
    print("Dataset shape:", df.shape)

    central_tendency = df.describe().loc[["mean", "50%", "std"]]
    central_tendency.rename(index={"50%": "median"}, inplace=True)
    print("\nOverall central tendency and dispersion (mean/median/std):")
    print(central_tendency)

    quality_means = df.groupby("quality")[
        [
            "alcohol",
            "volatile_acidity",
            "citric_acid",
            "sulphates",
            "total_sulfur_dioxide",
        ]
    ].mean()
    print("\nMean chemistry measurements by quality rating:")
    print(quality_means)

    correlations = df.corr(numeric_only=True)["quality"].sort_values(ascending=False)
    print("\nCorrelation of features with quality:")
    print(correlations)


if __name__ == "__main__":
    dataframe = load_dataset()
    summarize(dataframe)
