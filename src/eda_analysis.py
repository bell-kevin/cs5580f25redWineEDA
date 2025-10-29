"""Utilities for exploring the red wine quality dataset.

This module loads the dataset used in the accompanying exploratory data
analysis report and prints statistics that are incorporated into the
LaTeX document. In addition to the descriptive summaries that support the
existing tables, the module now provides reproducible outputs for two
additional analyses:

* a linear model capturing the interaction between volatile acidity and
  sulphates when predicting sensory quality, and
* a principal component analysis (PCA) of the physicochemical features
  that motivates the segmentation discussion.

The script only reads from the raw dataset so it remains safe to execute
in restricted environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "red_wine_quality.csv"


def load_dataset() -> pd.DataFrame:
    """Load the red wine quality dataset."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print key summary statistics for the dataset."""
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


def fit_acidity_sulfur_interaction(df: pd.DataFrame) -> Tuple[pd.Series, float, pd.DataFrame]:
    """Model the interaction between volatile acidity and sulphates."""

    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "volatile_acidity": df["volatile_acidity"],
            "sulphates": df["sulphates"],
            "interaction": df["volatile_acidity"] * df["sulphates"],
        }
    )
    response = df["quality"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design.to_numpy(), response, rcond=None)
    predictions = design.to_numpy() @ beta
    residuals = response - predictions
    total_sum_squares = np.sum((response - response.mean()) ** 2)
    residual_sum_squares = np.sum(residuals**2)
    r_squared = 1.0 - residual_sum_squares / total_sum_squares

    coefficients = pd.Series(beta, index=design.columns, name="coefficient")

    per_quality = []
    for quality, group in df.groupby("quality"):
        x = group["volatile_acidity"]
        y = group["total_sulfur_dioxide"]
        x_centered = x - x.mean()
        denominator = (x_centered**2).sum()
        slope = np.nan
        if denominator > 0:
            slope = float(((x_centered) * (y - y.mean())).sum() / denominator)
        per_quality.append(
            {
                "quality": quality,
                "va_to_so2_slope": slope,
                "va_so2_corr": x.corr(y),
                "mean_sulphates": group["sulphates"].mean(),
            }
        )

    interaction_summary = pd.DataFrame(per_quality).set_index("quality").sort_index()

    print(
        "\nLinear model coefficients for quality ~ volatile_acidity + sulphates + volatile_acidity*sulphates:"
    )
    print(coefficients)
    print(f"R-squared: {r_squared:.4f}")
    print("\nQuality-specific sulfur response to volatile acidity:")
    print(interaction_summary)

    return coefficients, r_squared, interaction_summary


def pca_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Compute PCA loadings and explained variance for the chemistry features."""

    features = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "ph",
        "sulphates",
        "alcohol",
    ]
    values = df[features].to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    standardized = (values - means) / stds
    covariance = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ordering = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:, ordering]
    explained_variance = eigenvalues / eigenvalues.sum()

    loadings = pd.DataFrame(eigenvectors[:, :3], index=features, columns=["PC1", "PC2", "PC3"])
    standardization = pd.DataFrame({"mean": means, "std": stds}, index=features)

    print("\nPCA loadings for the first three components:")
    print(loadings)
    print("\nExplained variance ratios (first five components):")
    explained = pd.Series(explained_variance[:5], index=[f"PC{i}" for i in range(1, 6)])
    print(explained)
    print("\nFeature means and standard deviations used for standardization:")
    print(standardization)

    return loadings, explained_variance, standardization


if __name__ == "__main__":
    dataframe = load_dataset()
    summarize(dataframe)
    fit_acidity_sulfur_interaction(dataframe)
    pca_summary(dataframe)
