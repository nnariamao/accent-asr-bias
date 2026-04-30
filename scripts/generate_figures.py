"""
generate_figures.py
--------------------
Generates four publication-quality figures (300 dpi PNG):
  fig1 - Scatter plot: accent strength vs WER (coloured by L1)
  fig2 - Box + strip plot: WER by L1 group
  fig3 - Bar chart: R² model comparison
  fig4 - Heatmap: inter-rater correlation matrix
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_FILE = os.path.join(BASE_DIR, "merged_analysis_multirater.csv")
RATINGS_RAW = os.path.join(BASE_DIR, "accent_rating_sheet_5raters.csv")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RATER_COLS  = ["Rater_1_Tim", "Rater_2", "Rater_3", "Rater_4", "Rater_5"]

# ── Colour palette (colourblind-friendly) ────────────────────────────────────
L1_PALETTE = {
    "Arabic":     "#E69F00",
    "Chinese":    "#56B4E9",
    "Hindi":      "#009E73",
    "Korean":     "#F0E442",
    "Spanish":    "#CC79A7",
    "Vietnamese": "#D55E00",
}

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      100,
})


# ────────────────────────────────────────────────────────────────────────────
# FIG 1  Scatter: Accent Strength vs WER  (coloured by L1)
# ────────────────────────────────────────────────────────────────────────────
def fig1_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    for l1, grp in df.groupby("L1_Background"):
        ax.scatter(grp["Mean_Accent_Strength"], grp["Mean_WER"],
                   color=L1_PALETTE[l1], s=90, zorder=3,
                   edgecolors="white", linewidths=0.6, label=l1)

    # Regression line
    x = df["Mean_Accent_Strength"].values
    y = df["Mean_WER"].values
    r, p = stats.pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    ax.plot(x_line, m * x_line + b, color="#333333",
            linewidth=1.8, linestyle="--", zorder=2, label="_nolegend_")

    sig_str = ("p < 0.001" if p < 0.001 else
               f"p < 0.01"  if p < 0.01  else
               f"p = {p:.3f}")
    ax.text(0.05, 0.94,
            f"r = {r:.3f},  {sig_str},  R² = {r**2:.3f}",
            transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Mean Accent Strength (1–5 scale, 5-rater average)")
    ax.set_ylabel("Word Error Rate (WER)")
    ax.set_title("Accent Strength vs ASR Word Error Rate\nby L1 Background")
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0.18, 0.60)

    legend = ax.legend(title="L1 Background", loc="lower right",
                       framealpha=0.9, edgecolor="#cccccc")
    legend.get_title().set_fontsize(10)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_scatter_accent_wer.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# FIG 2  Box + strip: WER by L1 group
# ────────────────────────────────────────────────────────────────────────────
def fig2_boxplot(df):
    order = df.groupby("L1_Background")["Mean_WER"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Box plot
    bp = ax.boxplot(
        [df.loc[df["L1_Background"] == l, "Mean_WER"].values for l in order],
        positions=range(len(order)),
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="", alpha=0),
    )
    for patch, l1 in zip(bp["boxes"], order):
        patch.set_facecolor(L1_PALETTE[l1])
        patch.set_alpha(0.55)

    # Individual speaker points
    np.random.seed(42)
    for i, l1 in enumerate(order):
        grp = df.loc[df["L1_Background"] == l1, "Mean_WER"].values
        jitter = np.random.uniform(-0.12, 0.12, size=len(grp))
        ax.scatter(i + jitter, grp,
                   color=L1_PALETTE[l1], s=70, zorder=4,
                   edgecolors="white", linewidths=0.6)

    # Annotate n per group
    for i, l1 in enumerate(order):
        n = (df["L1_Background"] == l1).sum()
        ax.text(i, 0.195, f"n={n}", ha="center", va="bottom",
                fontsize=9, color="#555555")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_xlabel("L1 Background")
    ax.set_ylabel("Word Error Rate (WER)")
    ax.set_title("Within-Group WER Variation by L1 Background\n"
                 "(boxes = IQR, points = individual speakers)")
    ax.set_ylim(0.18, 0.60)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_boxplot_within_group.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# FIG 3  Bar chart: R² model comparison
# ────────────────────────────────────────────────────────────────────────────
def fig3_model_comparison(r2_cat, r2_grad, r2_full):
    labels = [
        "Model 1\nWER ~ L1 Background\n(categorical)",
        "Model 2\nWER ~ Accent Strength\n(gradient)",
        "Model 3\nWER ~ L1 + Accent\n(full model)",
    ]
    values = [r2_cat, r2_grad, r2_full]
    colors = ["#56B4E9", "#E69F00", "#009E73"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(3), values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"R² = {val:.3f}\n({val*100:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, min(1.0, max(values) * 1.35))
    ax.set_ylabel("R² (Proportion of Variance Explained)")
    ax.set_title("Regression Model Comparison:\nPredicting ASR Word Error Rate")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_model_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# FIG 4  Heatmap: Inter-rater correlation matrix
# ────────────────────────────────────────────────────────────────────────────
def fig4_heatmap(raw_df):
    corr = raw_df[RATER_COLS].corr()

    # Prettier labels
    nice_labels = ["Rater 1\n(Tim)", "Rater 2", "Rater 3", "Rater 4", "Rater 5"]
    corr.index   = nice_labels
    corr.columns = nice_labels

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr, dtype=bool)

    sns.heatmap(corr, annot=True, fmt=".3f",
                cmap="YlOrRd", vmin=0.5, vmax=1.0,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Pearson r", "shrink": 0.85},
                annot_kws={"size": 11})

    ax.set_title("Inter-Rater Correlation Matrix\n(Accent Strength Ratings, N = 24 speakers)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_interrater_correlation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 3: GENERATE FIGURES")
    print("=" * 60)

    df     = pd.read_csv(MERGED_FILE)
    raw_df = pd.read_csv(RATINGS_RAW)

    # Rebuild model R² values inline (avoids importing the analysis script)
    accent = df["Mean_Accent_Strength"].values
    wer_v  = df["Mean_WER"].values
    l1_dummies = pd.get_dummies(df["L1_Background"], drop_first=True).values

    def fit_r2(X, y):
        return LinearRegression().fit(X, y).score(X, y)

    r2_cat  = fit_r2(l1_dummies, wer_v)
    r2_grad = fit_r2(accent.reshape(-1, 1), wer_v)
    r2_full = fit_r2(np.hstack([l1_dummies, accent.reshape(-1, 1)]), wer_v)

    print(f"\nGenerating figures in: {FIG_DIR}")

    print("\n[1/4] Scatter plot (accent vs WER)...")
    fig1_scatter(df)

    print("[2/4] Box plot (WER by L1 group)...")
    fig2_boxplot(df)

    print("[3/4] Model comparison bar chart...")
    fig3_model_comparison(r2_cat, r2_grad, r2_full)

    print("[4/4] Inter-rater correlation heatmap...")
    fig4_heatmap(raw_df)

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE  –  4 figures saved")
    print("=" * 60)


if __name__ == "__main__":
    main()
