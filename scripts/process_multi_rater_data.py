"""
process_multi_rater_data.py
----------------------------
Loads the completed 5-rater accent rating sheet, computes per-speaker
mean and standard deviation across raters, calculates Cronbach's Alpha
for inter-rater reliability, and saves the processed data.
"""

import os
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "accent_rating_sheet_5raters.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "accent_ratings_5raters.csv")

RATER_COLS = ["Rater_1_Tim", "Rater_2", "Rater_3", "Rater_4", "Rater_5"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def cronbach_alpha(df_raters: pd.DataFrame) -> float:
    """
    Compute Cronbach's Alpha for a DataFrame where each column is one rater.
    Formula: alpha = (k / (k-1)) * (1 - sum(item_variances) / total_variance)
    """
    k = df_raters.shape[1]
    item_variances = df_raters.var(axis=0, ddof=1).sum()
    total_variance = df_raters.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_variances / total_variance)
    return alpha


def interpret_alpha(alpha: float) -> str:
    if alpha >= 0.90:
        return "Excellent"
    elif alpha >= 0.80:
        return "Good"
    elif alpha >= 0.70:
        return "Acceptable"
    elif alpha >= 0.60:
        return "Questionable"
    elif alpha >= 0.50:
        return "Poor"
    else:
        return "Unacceptable"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1: PROCESS MULTI-RATER DATA")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} speakers, columns: {list(df.columns)}")

    # Check for missing rater values
    missing = df[RATER_COLS].isnull().sum()
    if missing.any():
        print(f"\n  WARNING: Missing values detected:\n{missing[missing > 0]}")
    else:
        print("  No missing values in rater columns.")

    # Compute mean and std per speaker
    df["Mean_Accent_Strength"] = df[RATER_COLS].mean(axis=1)
    df["Std_Accent_Strength"]  = df[RATER_COLS].std(axis=1, ddof=1)

    # Build output dataframe
    out = df[["Speaker_ID", "L1_Background",
              "Mean_Accent_Strength", "Std_Accent_Strength"]].copy()

    # ── Cronbach's Alpha ──────────────────────────────────────────────────────
    print("\n--- Inter-Rater Reliability (Cronbach's Alpha) ---")
    alpha = cronbach_alpha(df[RATER_COLS])
    interp = interpret_alpha(alpha)
    print(f"  Cronbach's Alpha = {alpha:.4f}  ({interp})")
    print(f"  N raters = {len(RATER_COLS)},  N speakers = {len(df)}")

    # ── Pairwise rater correlations ───────────────────────────────────────────
    print("\n--- Pairwise Rater Correlations ---")
    corr_matrix = df[RATER_COLS].corr()
    pairs = []
    for i in range(len(RATER_COLS)):
        for j in range(i + 1, len(RATER_COLS)):
            r = corr_matrix.iloc[i, j]
            pairs.append((RATER_COLS[i], RATER_COLS[j], r))
            print(f"  {RATER_COLS[i]} vs {RATER_COLS[j]}: r = {r:.3f}")
    avg_r = np.mean([p[2] for p in pairs])
    print(f"  Mean pairwise r = {avg_r:.3f}")

    # ── Rating variance summary ───────────────────────────────────────────────
    print("\n--- Speaker Rating Variance ---")
    sorted_by_std = out.sort_values("Std_Accent_Strength", ascending=False)

    print("\n  Highest variance speakers (most disagreement):")
    for _, row in sorted_by_std.head(5).iterrows():
        print(f"    {row['Speaker_ID']:6s} ({row['L1_Background']:12s})"
              f"  mean={row['Mean_Accent_Strength']:.2f}"
              f"  std={row['Std_Accent_Strength']:.2f}")

    print("\n  Lowest variance speakers (most agreement):")
    for _, row in sorted_by_std.tail(5).iterrows():
        print(f"    {row['Speaker_ID']:6s} ({row['L1_Background']:12s})"
              f"  mean={row['Mean_Accent_Strength']:.2f}"
              f"  std={row['Std_Accent_Strength']:.2f}")

    # ── Per-L1 summary ────────────────────────────────────────────────────────
    print("\n--- Per-L1 Mean Accent Strength ---")
    l1_summary = out.groupby("L1_Background")["Mean_Accent_Strength"].agg(
        ["mean", "std", "count"]).round(3)
    print(l1_summary.to_string())

    # ── Overall distribution ──────────────────────────────────────────────────
    print(f"\n--- Overall Accent Strength Distribution ---")
    print(f"  Min:    {out['Mean_Accent_Strength'].min():.2f}")
    print(f"  Max:    {out['Mean_Accent_Strength'].max():.2f}")
    print(f"  Mean:   {out['Mean_Accent_Strength'].mean():.2f}")
    print(f"  Median: {out['Mean_Accent_Strength'].median():.2f}")
    print(f"  SD:     {out['Mean_Accent_Strength'].std():.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_sorted = out.sort_values("Speaker_ID").reset_index(drop=True)
    out_sorted.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  Columns: {list(out_sorted.columns)}")
    print(f"  Rows:    {len(out_sorted)}")

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)

    # Return key stats for use by summary report
    return {
        "cronbach_alpha": alpha,
        "alpha_interpretation": interp,
        "avg_pairwise_r": avg_r,
        "n_speakers": len(df),
        "n_raters": len(RATER_COLS),
    }


if __name__ == "__main__":
    main()
