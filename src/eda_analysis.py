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
from typing import Iterable, Tuple

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


def _initialize_centroids(values: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministically initialize centroids using k-means++ style seeding."""

    rng = np.random.default_rng(seed)
    centroids = np.empty((k, values.shape[1]), dtype=float)
    # Choose the first centroid uniformly at random
    first_index = rng.integers(0, values.shape[0])
    centroids[0] = values[first_index]

    # Subsequent centroids follow k-means++ weighting
    distances = np.linalg.norm(values - centroids[0], axis=1) ** 2
    for idx in range(1, k):
        probabilities = distances / distances.sum()
        chosen_index = rng.choice(values.shape[0], p=probabilities)
        centroids[idx] = values[chosen_index]
        new_distances = np.linalg.norm(values - centroids[idx], axis=1) ** 2
        distances = np.minimum(distances, new_distances)

    return centroids


def _kmeans(
    values: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run Lloyd's algorithm and return labels, centroids, and inertia."""

    centroids = _initialize_centroids(values, k, seed=seed)
    for _ in range(max_iter):
        # Assignment step
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Update step
        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = values[labels == cluster_id]
            if len(members) > 0:
                new_centroids[cluster_id] = members.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift <= tol:
            break

    final_distances = np.linalg.norm(values - centroids[labels], axis=1) ** 2
    inertia = float(final_distances.sum())
    return labels, centroids, inertia


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean silhouette score for the provided clustering."""

    n_samples = values.shape[0]
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    silhouettes = np.zeros(n_samples)
    unique_labels = np.unique(labels)

    for i in range(n_samples):
        own_label = labels[i]
        same_mask = labels == own_label
        same_count = same_mask.sum()

        # Intra-cluster distance
        if same_count > 1:
            a = distances[i, same_mask].sum() / (same_count - 1)
        else:
            a = 0.0

        # Nearest other cluster
        b = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = labels == other_label
            if not np.any(other_mask):
                continue
            mean_distance = distances[i, other_mask].mean()
            if mean_distance < b:
                b = mean_distance

        if np.isinf(b) and a == 0.0:
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b - a) / max(a, b)

    return float(silhouettes.mean())


def clustering_diagnostics(
    df: pd.DataFrame,
    cluster_counts: Iterable[int] = (2, 3, 4, 5),
    *,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate k-means clustering for multiple cluster counts."""

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
    standardized = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)

    evaluations = []
    chosen_profiles = None

    for k in cluster_counts:
        labels, _, inertia = _kmeans(standardized, k, seed=seed)
        silhouette = _silhouette_score(standardized, labels)
        evaluations.append({"clusters": k, "silhouette": silhouette, "inertia": inertia})

        if chosen_profiles is None or silhouette > chosen_profiles[0]:
            summary = (
                df.assign(cluster=labels)[
                    [
                        "cluster",
                        "quality",
                        "alcohol",
                        "volatile_acidity",
                        "sulphates",
                    ]
                ]
                .groupby("cluster")
                .agg(
                    size=("quality", "size"),
                    mean_quality=("quality", "mean"),
                    mean_alcohol=("alcohol", "mean"),
                    mean_volatile_acidity=("volatile_acidity", "mean"),
                    mean_sulphates=("sulphates", "mean"),
                )
                .reset_index()
                .sort_values("mean_quality", ascending=False)
            )
            chosen_profiles = (silhouette, k, summary)

    evaluations_df = pd.DataFrame(evaluations).set_index("clusters").sort_index()
    best_summary = chosen_profiles[2].set_index("cluster")

    print("\nK-means evaluation across cluster counts:")
    print(evaluations_df)
    print(
        f"\nBest-performing clustering: k={chosen_profiles[1]} with silhouette={chosen_profiles[0]:.3f}."
    )
    print("Cluster profiles (size and key chemistry means):")
    print(best_summary)

    return evaluations_df, best_summary


if __name__ == "__main__":
    dataframe = load_dataset()
    summarize(dataframe)
    fit_acidity_sulfur_interaction(dataframe)
    pca_summary(dataframe)
    clustering_diagnostics(dataframe)
