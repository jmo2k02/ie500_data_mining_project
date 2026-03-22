"""
Visualization functions.

Every public function takes a DataFrame + results_dir and saves a PNG.
Matplotlib is configured for non-interactive (Agg) use.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

from log import log, step_timer

# Global style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150

# ── Language → country mapping ───────────────────────────────────────
# Maps Twitch ISO-639-1 language codes to lists of ADM0_A3 country codes.
# Every country that speaks a language receives the full user count for
# that language (no weighting), since the dataset only records the stream
# language, not the streamer's actual country.  When a country appears
# under multiple languages the counts are summed (e.g. Canada = EN + FR,
# Belgium = FR + NL, Switzerland = DE + FR).
_LANG_TO_COUNTRIES: dict[str, list[str]] = {
    "EN": ["USA", "GBR", "CAN", "AUS", "IRL", "NZL", "ZAF"],
    "DE": ["DEU", "AUT", "CHE"],
    "FR": ["FRA", "BEL", "CHE", "CAN"],
    "ES": ["ESP", "MEX", "ARG", "COL", "CHL", "PER", "VEN", "ECU", "BOL"],
    "RU": ["RUS", "UKR", "BLR"],
    "ZH": ["CHN", "TWN", "SGP"],
    "PT": ["BRA", "PRT"],
    "JA": ["JPN"],
    "IT": ["ITA"],
    "KO": ["KOR"],
    "PL": ["POL"],
    "SV": ["SWE"],
    "TR": ["TUR"],
    "NL": ["NLD", "BEL"],
    "FI": ["FIN"],
    "TH": ["THA"],
    "CS": ["CZE"],
    "DA": ["DNK"],
    "HU": ["HUN"],
    "NO": ["NOR"],
    # OTHER is intentionally omitted – no single country to assign it to
}


def _save(fig, results_dir: str, filename: str) -> None:
    path = os.path.join(results_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    -> {filename}")


# ── Pre-network plots ────────────────────────────────────────────────


def plot_binary_features(features_df: pd.DataFrame, results_dir: str) -> None:
    cols = ["explicit_content", "dead_account", "affiliate_status"]
    with step_timer("Binary features bar chart"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, col in enumerate(cols):
            if col not in features_df.columns:
                continue
            counts = features_df[col].value_counts().sort_index()
            counts.plot(kind="bar", ax=axes[idx], color=["#1f77b4", "#ff7f0e"])
            axes[idx].set_title(
                f"{col.replace('_', ' ').title()} Distribution",
                fontsize=12,
                fontweight="bold",
            )
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel("Count")
            axes[idx].tick_params(axis="x", rotation=0)
            axes[idx].set_xticklabels(["No (0)", "Yes (1)"])
            for container in axes[idx].containers:
                axes[idx].bar_label(container, fmt="%d", fontsize=9)
        fig.tight_layout()
        _save(fig, results_dir, "binary_features_distribution.png")


def plot_language(features_df: pd.DataFrame, results_dir: str) -> None:
    if "language" not in features_df.columns:
        return
    with step_timer("Language distribution"):
        fig, ax = plt.subplots(figsize=(12, 7))
        top = features_df["language"].value_counts().head(15)
        ax.barh(range(len(top)), top.values, color="steelblue")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index)
        ax.set_title("Top 15 Languages on Twitch", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Users")
        ax.set_ylabel("Language")
        ax.invert_yaxis()
        for i, v in enumerate(top.values):
            ax.text(v + 500, i, f"{v:,}", va="center", fontsize=9)
        fig.tight_layout()
        _save(fig, results_dir, "language_distribution.png")


def plot_language_worldmap(features_df: pd.DataFrame, results_dir: str) -> None:
    """Plot a world choropleth map coloured by Twitch user count per country.

    Language codes are mapped to countries using ``_LANG_TO_COUNTRIES``.
    Countries with no users remain light grey; countries with users are
    coloured on a light-green → dark-red gradient.
    """
    if "language" not in features_df.columns:
        return
    with step_timer("Language world map"):
        # --- aggregate user counts per country (ADM0_A3) ----------------
        # Each country that speaks a language receives the *full* user
        # count for that language.  When a country appears under multiple
        # languages the counts are summed (e.g. Canada = EN + FR).
        lang_counts = features_df["language"].value_counts()
        country_users: dict[str, float] = {}
        for lang, count in lang_counts.items():
            if lang not in _LANG_TO_COUNTRIES:
                continue
            for adm0 in _LANG_TO_COUNTRIES[lang]:
                country_users[adm0] = country_users.get(adm0, 0.0) + count

        if not country_users:
            log.info("    Skipped (no mappable language data)")
            return

        # --- load world geometry ----------------------------------------
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        )
        world["users"] = world["ADM0_A3"].map(country_users).fillna(0)

        # --- custom colour map: light green → yellow → dark red ---------
        cmap = LinearSegmentedColormap.from_list(
            "green_red",
            ["#d4edda", "#ffc107", "#dc3545", "#7a0012"],
            N=256,
        )

        # --- plot -------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(18, 9))
        # base layer: all countries in light grey
        world.plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.3)
        # overlay: only countries with users > 0
        has_users = world[world["users"] > 0].copy()
        if len(has_users) > 0:
            has_users.plot(
                column="users",
                ax=ax,
                cmap=cmap,
                edgecolor="white",
                linewidth=0.3,
                legend=True,
                legend_kwds={
                    "label": "Number of Twitch Users",
                    "orientation": "horizontal",
                    "shrink": 0.6,
                    "pad": 0.05,
                },
            )
            # ── One label per language, with lines to every country ──
            # Build a lookup from ADM0_A3 → centroid for labelled countries
            adm0_centroid: dict[str, tuple[float, float]] = {}
            for _, row in has_users.iterrows():
                c = row.geometry.centroid
                adm0_centroid[row["ADM0_A3"]] = (c.x, c.y)

            # Where to place each language label (tx, ty) in data coords.
            # Tuned manually so labels don't overlap each other.
            _label_pos: dict[str, tuple[float, float]] = {
                "EN": (-40, -30),
                "DE": (42, 62),
                "FR": (-30, 18),
                "ES": (-100, -18),
                "RU": (100, 72),
                "ZH": (135, 18),
                "PT": (-60, -40),
                "JA": (160, 42),
                "IT": (20, 28),
                "KO": (150, 30),
                "PL": (38, 52),
                "SV": (-8, 72),
                "TR": (52, 28),
                "NL": (0, 55),
                "FI": (55, 72),
                "TH": (115, 6),
                "CS": (28, 42),
                "DA": (8, 65),
                "HU": (35, 36),
                "NO": (-18, 65),
            }

            for lang, adm0_list in _LANG_TO_COUNTRIES.items():
                if lang not in lang_counts.index:
                    continue
                count = lang_counts[lang]
                if lang not in _label_pos:
                    continue

                tx, ty = _label_pos[lang]

                # collect centroids of all countries for this language
                targets = [adm0_centroid[a] for a in adm0_list if a in adm0_centroid]
                if not targets:
                    continue

                # draw leader lines from label to every country centroid
                for cx, cy in targets:
                    ax.annotate(
                        "",
                        xy=(cx, cy),
                        xytext=(tx, ty),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="#555555",
                            lw=0.8,
                            shrinkA=12,
                            shrinkB=2,
                        ),
                    )

                # draw the label itself
                ax.annotate(
                    f"{lang}: {count:,}",
                    xy=(tx, ty),
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="#888888",
                        linewidth=0.5,
                    ),
                )

        ax.set_title(
            "Twitch Users by Country (based on stream language)",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, results_dir, "language_worldmap.png")


def plot_views(features_df: pd.DataFrame, results_dir: str) -> None:
    if "view_count" not in features_df.columns:
        return
    with step_timer("Views distribution"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].hist(
            features_df["view_count"],
            bins=50,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_title("Views Distribution (Linear)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Views")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(
            np.log10(features_df["view_count"] + 1),
            bins=50,
            color="coral",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_title("Views Distribution (Log10)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Log10(Views + 1)")
        axes[1].set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, results_dir, "views_distribution.png")


def plot_lifetime(features_df: pd.DataFrame, results_dir: str) -> None:
    if "account_lifetime" not in features_df.columns:
        return
    with step_timer("Account lifetime distribution"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].hist(
            features_df["account_lifetime"],
            bins=50,
            color="green",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_title(
            "Account Lifetime Distribution (Linear)", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Account Lifetime (days)")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(
            np.log10(features_df["account_lifetime"] + 1),
            bins=50,
            color="purple",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_title(
            "Account Lifetime Distribution (Log10)", fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Log10(Account Lifetime + 1)")
        axes[1].set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, results_dir, "lifetime_distribution.png")


def plot_correlation(
    features_df: pd.DataFrame,
    results_dir: str,
    filename: str = "correlation_matrix.png",
) -> None:
    with step_timer("Correlation heatmap"):
        num = features_df.select_dtypes(include=[np.number])
        if len(num.columns) < 2:
            log.info("    Skipped (fewer than 2 numeric columns)")
            return
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = num.corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 8},
            ax=ax,
        )
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, results_dir, filename)


# ── Post-network plots ───────────────────────────────────────────────


def plot_degree(features_df: pd.DataFrame, results_dir: str) -> None:
    if "degree" not in features_df.columns:
        return
    with step_timer("Degree distribution plot"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].hist(
            features_df["degree"],
            bins=50,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_title(
            "Degree Distribution (Linear)", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Degree")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(
            np.log10(features_df["degree"] + 1),
            bins=50,
            color="coral",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_title("Degree Distribution (Log10)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Log10(Degree + 1)")
        axes[1].set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, results_dir, "degree_distribution.png")


def plot_clustering(features_df: pd.DataFrame, results_dir: str) -> None:
    if "clustering_coeff" not in features_df.columns:
        return
    with step_timer("Clustering coefficient plot"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            features_df["clustering_coeff"],
            bins=50,
            color="green",
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(
            "Clustering Coefficient Distribution", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Clustering Coefficient")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, results_dir, "clustering_coefficient_distribution.png")


def plot_pagerank(features_df: pd.DataFrame, results_dir: str) -> None:
    if "pagerank" not in features_df.columns:
        return
    with step_timer("PageRank distribution plot"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].hist(
            features_df["pagerank"],
            bins=50,
            color="purple",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_title(
            "PageRank Distribution (Linear)", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("PageRank")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(
            np.log10(features_df["pagerank"] + 1e-10),
            bins=50,
            color="orange",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_title(
            "PageRank Distribution (Log10)", fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Log10(PageRank)")
        axes[1].set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, results_dir, "pagerank_distribution.png")


def plot_core_number(features_df: pd.DataFrame, results_dir: str) -> None:
    if "core_number" not in features_df.columns:
        return
    with step_timer("Core number distribution plot"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            features_df["core_number"],
            bins=50,
            color="teal",
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(
            "Core Number Distribution (k-core)", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Core Number")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, results_dir, "core_number_distribution.png")


# ── Convenience runners ──────────────────────────────────────────────


def plot_all_basic(features_df: pd.DataFrame, results_dir: str) -> None:
    """Generate all pre-network visualizations."""
    plot_binary_features(features_df, results_dir)
    plot_language(features_df, results_dir)
    plot_language_worldmap(features_df, results_dir)
    plot_views(features_df, results_dir)
    plot_lifetime(features_df, results_dir)
    plot_correlation(features_df, results_dir)


def plot_all_network(features_df: pd.DataFrame, results_dir: str) -> None:
    """Generate all network-feature visualizations + updated correlation."""
    plot_degree(features_df, results_dir)
    plot_clustering(features_df, results_dir)
    plot_pagerank(features_df, results_dir)
    plot_core_number(features_df, results_dir)
    plot_correlation(features_df, results_dir, "correlation_matrix_with_network.png")
