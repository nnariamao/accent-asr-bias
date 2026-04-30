"""
final_analysis_multirater.py
-----------------------------
Merges accent ratings with ASR WER data and runs four analyses:
  1. Pearson correlation: Mean_Accent_Strength vs Mean_WER
  2. One-way ANOVA: L1_Background -> Mean_WER
  3. R² comparison of three regression models
  4. Within-group analysis per L1
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATINGS_FILE   = os.path.join(BASE_DIR, "accent_ratings_5raters.csv")
WER_FILE       = os.path.join(BASE_DIR, "speaker_wer_summary.csv")
MERGED_OUT     = os.path.join(BASE_DIR, "merged_analysis_multirater.csv")
WITHIN_OUT     = os.path.join(BASE_DIR, "within_group_multirater.csv")
STATS_JSON     = os.path.join(BASE_DIR, "stats_results.json")   # for summary report


# ── Helper: R² from sklearn LinearRegression ─────────────────────────────────
def fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    model = LinearRegression().fit(X, y)
    return model.score(X, y)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 2: STATISTICAL ANALYSIS (MULTI-RATER)")
    print("=" * 60)

    # ── Load & merge ──────────────────────────────────────────────────────────
    print(f"\nLoading ratings: {RATINGS_FILE}")
    ratings = pd.read_csv(RATINGS_FILE)

    print(f"Loading WER data: {WER_FILE}")
    wer = pd.read_csv(WER_FILE)

    # Merge on Speaker_ID; keep L1 from ratings (authoritative)
    merged = pd.merge(
        ratings[["Speaker_ID", "L1_Background",
                 "Mean_Accent_Strength", "Std_Accent_Strength"]],
        wer[["Speaker_ID", "Mean_WER", "Std_WER", "Num_Files"]],
        on="Speaker_ID",
        how="inner"
    )
    print(f"  Merged: {len(merged)} speakers")

    if len(merged) < len(ratings):
        missing = set(ratings.Speaker_ID) - set(wer.Speaker_ID)
        print(f"  WARNING: {len(missing)} speakers missing from WER file: {missing}")

    merged.to_csv(MERGED_OUT, index=False)
    print(f"  Saved merged data: {MERGED_OUT}")

    # Shorthand
    accent = merged["Mean_Accent_Strength"].values
    wer_v  = merged["Mean_WER"].values
    l1     = merged["L1_Background"].values

    # ── ANALYSIS 1: Pearson Correlation ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("ANALYSIS 1: Pearson Correlation (Accent Strength vs WER)")
    print("─" * 60)

    r, p_corr = stats.pearsonr(accent, wer_v)
    r2_corr   = r ** 2
    n         = len(merged)

    # 95% confidence interval via Fisher's z
    z    = np.arctanh(r)
    se   = 1 / np.sqrt(n - 3)
    ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

    print(f"  r  = {r:.4f}")
    print(f"  p  = {p_corr:.6f}  {'***' if p_corr < 0.001 else ('**' if p_corr < 0.01 else ('*' if p_corr < 0.05 else 'n.s.'))}")
    print(f"  R² = {r2_corr:.4f}  (explains {r2_corr*100:.1f}% of WER variance)")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  N  = {n}")

    # ── ANALYSIS 2: One-Way ANOVA (L1 -> WER) ────────────────────────────────
    print("\n" + "─" * 60)
    print("ANALYSIS 2: One-Way ANOVA (L1_Background -> Mean_WER)")
    print("─" * 60)

    groups = [merged.loc[merged["L1_Background"] == g, "Mean_WER"].values
              for g in merged["L1_Background"].unique()]
    F, p_anova = stats.f_oneway(*groups)

    # Effect size eta-squared
    grand_mean  = wer_v.mean()
    ss_between  = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total    = sum((x - grand_mean) ** 2 for x in wer_v)
    eta_sq      = ss_between / ss_total

    print(f"  F({len(groups)-1}, {n - len(groups)}) = {F:.4f}")
    print(f"  p  = {p_anova:.6f}  {'***' if p_anova < 0.001 else ('**' if p_anova < 0.01 else ('*' if p_anova < 0.05 else 'n.s.'))}")
    print(f"  η² = {eta_sq:.4f}  (explains {eta_sq*100:.1f}% of WER variance)")

    print("\n  Group means:")
    for l1_name in sorted(merged["L1_Background"].unique()):
        grp = merged.loc[merged["L1_Background"] == l1_name, "Mean_WER"]
        print(f"    {l1_name:12s}: mean={grp.mean():.4f}  sd={grp.std():.4f}  n={len(grp)}")

    # ── ANALYSIS 3: Model R² Comparison ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("ANALYSIS 3: R² Comparison of Three Regression Models")
    print("─" * 60)

    y = wer_v.reshape(-1, 1)

    # Model 1: WER ~ L1_Background (dummy variables)
    l1_dummies = pd.get_dummies(merged["L1_Background"], drop_first=True).values
    r2_m1 = fit_r2(l1_dummies, wer_v)

    # Model 2: WER ~ Mean_Accent_Strength (gradient)
    r2_m2 = fit_r2(accent.reshape(-1, 1), wer_v)

    # Model 3: WER ~ L1_Background + Mean_Accent_Strength (full)
    X_full = np.hstack([l1_dummies, accent.reshape(-1, 1)])
    r2_m3  = fit_r2(X_full, wer_v)

    print(f"  Model 1  WER ~ L1_Background              R² = {r2_m1:.4f}  ({r2_m1*100:.1f}%)")
    print(f"  Model 2  WER ~ Mean_Accent_Strength        R² = {r2_m2:.4f}  ({r2_m2*100:.1f}%)")
    print(f"  Model 3  WER ~ L1 + Mean_Accent_Strength   R² = {r2_m3:.4f}  ({r2_m3*100:.1f}%)")

    gradient_wins = r2_m2 > r2_m1
    delta = r2_m2 - r2_m1
    print(f"\n  Gradient vs Categorical: ΔR² = {delta:+.4f}")
    if gradient_wins:
        print(f"  → Gradient (accent strength) explains MORE variance than categorical L1")
    else:
        print(f"  → Categorical (L1 group) explains MORE variance than gradient accent strength")
    print(f"  → Full model (both) R² = {r2_m3:.4f} (incremental gain over best single: {r2_m3 - max(r2_m1, r2_m2):+.4f})")

    # ── ANALYSIS 4: Within-Group Analysis ────────────────────────────────────
    print("\n" + "─" * 60)
    print("ANALYSIS 4: Within-Group Analysis by L1 Background")
    print("─" * 60)

    within_rows = []
    for l1_name in sorted(merged["L1_Background"].unique()):
        grp = merged[merged["L1_Background"] == l1_name].copy()
        wer_range     = grp["Mean_WER"].max() - grp["Mean_WER"].min()
        accent_range  = grp["Mean_Accent_Strength"].max() - grp["Mean_Accent_Strength"].min()
        n_grp         = len(grp)

        if n_grp > 2:
            r_within, p_within = stats.pearsonr(grp["Mean_Accent_Strength"],
                                                grp["Mean_WER"])
        else:
            r_within, p_within = np.nan, np.nan

        sig_str = ('***' if p_within < 0.001 else
                   ('**'  if p_within < 0.01  else
                    ('*'   if p_within < 0.05  else 'n.s.'))) if not np.isnan(p_within) else "n/a"

        print(f"\n  {l1_name} (n={n_grp}):")
        print(f"    WER range:    {grp['Mean_WER'].min():.4f} – {grp['Mean_WER'].max():.4f}  (Δ={wer_range:.4f})")
        print(f"    Accent range: {grp['Mean_Accent_Strength'].min():.2f} – {grp['Mean_Accent_Strength'].max():.2f}  (Δ={accent_range:.2f})")
        print(f"    Within-group r = {r_within:.4f}  p = {p_within:.4f}  {sig_str}" if not np.isnan(r_within)
              else f"    Within-group r: insufficient data")

        within_rows.append({
            "L1_Background":    l1_name,
            "N_Speakers":       n_grp,
            "WER_Min":          grp["Mean_WER"].min(),
            "WER_Max":          grp["Mean_WER"].max(),
            "WER_Range":        wer_range,
            "WER_Mean":         grp["Mean_WER"].mean(),
            "Accent_Min":       grp["Mean_Accent_Strength"].min(),
            "Accent_Max":       grp["Mean_Accent_Strength"].max(),
            "Accent_Range":     accent_range,
            "Accent_Mean":      grp["Mean_Accent_Strength"].mean(),
            "Within_r":         round(r_within, 4) if not np.isnan(r_within) else None,
            "Within_p":         round(p_within, 4) if not np.isnan(p_within) else None,
        })

    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(WITHIN_OUT, index=False)
    print(f"\nSaved within-group results: {WITHIN_OUT}")

    # ── Save stats for summary report ─────────────────────────────────────────
    stats_out = {
        "correlation": {"r": round(r, 4), "p": round(p_corr, 6),
                        "r2": round(r2_corr, 4), "ci": [round(ci_lo, 4), round(ci_hi, 4)]},
        "anova":       {"F": round(F, 4), "p": round(p_anova, 6), "eta_sq": round(eta_sq, 4)},
        "models":      {"r2_categorical": round(r2_m1, 4),
                        "r2_gradient":    round(r2_m2, 4),
                        "r2_full":        round(r2_m3, 4),
                        "gradient_wins":  bool(gradient_wins),
                        "delta_r2":       round(delta, 4)},
        "n_speakers":  int(n),
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Saved stats JSON: {STATS_JSON}")

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)

    return stats_out


if __name__ == "__main__":
    main()
