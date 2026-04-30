import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("="*60)
print("Statistical Analysis: Categorical vs Gradient Accent")
print("="*60)

# Load data
wer_summary = pd.read_csv("/Users/timnariamao/accent_bias_project/speaker_wer_summary.csv")
accent_ratings = pd.read_csv("/Users/timnariamao/accent_bias_project/accent_ratings.csv")

# Merge datasets
df = pd.merge(wer_summary, accent_ratings, on=['Speaker_ID', 'L1_Background'])
print(f"\n✓ Merged dataset: {len(df)} speakers")
print(df[['Speaker_ID', 'L1_Background', 'Mean_WER', 'Accent_Strength']])

# ============================================================
# ANALYSIS 1: Correlation between Accent Strength and WER
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: Correlation (Gradient Accent vs WER)")
print("="*60)

correlation, p_value = stats.pearsonr(df['Accent_Strength'], df['Mean_WER'])
print(f"\nPearson Correlation (r): {correlation:.3f}")
print(f"P-value:                 {p_value:.4f}")
print(f"R-squared (r²):          {correlation**2:.3f}")

if p_value < 0.05:
    print(f"\n✓ SIGNIFICANT: Accent strength significantly predicts WER (p < 0.05)")
else:
    print(f"\n✗ NOT SIGNIFICANT: p = {p_value:.4f}")

# ============================================================
# ANALYSIS 2: ANOVA - Does L1 Category predict WER?
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 2: ANOVA (Categorical L1 vs WER)")
print("="*60)

# Group WERs by L1 background
l1_groups = [group['Mean_WER'].values for name, group in df.groupby('L1_Background')]
f_stat, p_anova = stats.f_oneway(*l1_groups)

print(f"\nOne-way ANOVA:")
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value:     {p_anova:.4f}")

if p_anova < 0.05:
    print(f"\n✓ SIGNIFICANT: L1 category significantly predicts WER (p < 0.05)")
else:
    print(f"\n✗ NOT SIGNIFICANT: p = {p_anova:.4f}")

# ============================================================
# ANALYSIS 3: R² Comparison
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 3: R² Comparison (Key Finding!)")
print("="*60)

# R² for gradient accent strength
r2_gradient = correlation**2
print(f"\nGradient Model (Accent Strength):")
print(f"  R² = {r2_gradient:.3f} ({r2_gradient*100:.1f}% variance explained)")

# R² for categorical L1 (using dummy variables)
l1_dummies = pd.get_dummies(df['L1_Background'], drop_first=True)

model_categorical = LinearRegression()
model_categorical.fit(l1_dummies, df['Mean_WER'])
r2_categorical = r2_score(df['Mean_WER'], model_categorical.predict(l1_dummies))
print(f"\nCategorical Model (L1 Background):")
print(f"  R² = {r2_categorical:.3f} ({r2_categorical*100:.1f}% variance explained)")

print(f"\n{'='*60}")
print(f"KEY FINDING:")
if r2_gradient > r2_categorical:
    print(f"✓ Gradient accent strength explains MORE variance than categorical L1!")
    print(f"  Gradient R²:     {r2_gradient:.3f}")
    print(f"  Categorical R²:  {r2_categorical:.3f}")
    print(f"  Improvement:     +{(r2_gradient - r2_categorical):.3f}")
else:
    print(f"Categorical L1 explains more variance than gradient accent strength")
    print(f"  Gradient R²:     {r2_gradient:.3f}")
    print(f"  Categorical R²:  {r2_categorical:.3f}")

# ============================================================
# ANALYSIS 4: Within-Group Variation
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 4: Within-Group Variation")
print("="*60)

print("\nWER Range Within Each L1 Group:")
for l1, group in df.groupby('L1_Background'):
    wer_range = group['Mean_WER'].max() - group['Mean_WER'].min()
    accent_range = group['Accent_Strength'].max() - group['Accent_Strength'].min()
    print(f"\n{l1}:")
    print(f"  WER range:            {wer_range:.3f} ({wer_range*100:.1f}%)")
    print(f"  Accent strength range: {accent_range}")
    print(f"  Speakers:             {group['Speaker_ID'].tolist()}")
    for _, row in group.iterrows():
        print(f"    {row['Speaker_ID']}: WER={row['Mean_WER']:.3f}, Accent={row['Accent_Strength']}")

# ============================================================
# SAVE MERGED DATASET
# ============================================================
df.to_csv("/Users/timnariamao/accent_bias_project/merged_analysis.csv", index=False)
print("\n" + "="*60)
print("✓ Merged analysis saved to: merged_analysis.csv")
print("="*60)