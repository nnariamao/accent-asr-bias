import pandas as pd

print("="*60)
print("Generating Final Rating Sheet for 5 Raters")
print("="*60)

# Load Rater 1 ratings
my_ratings = pd.read_csv("accent_ratings.csv")

# Load sample info for audio filenames
sample_info = pd.read_csv("/Users/timnariamao/accent_bias_project/rating_samples_info.csv")

# Merge to get audio filenames
df = pd.merge(my_ratings, sample_info[['Speaker_ID', 'Sample_File']], on='Speaker_ID')

# Sort by Speaker_ID for easy reference
df = df.sort_values('Speaker_ID').reset_index(drop=True)

# Create rating sheet with Rater_1 pre-filled
rating_sheet = pd.DataFrame({
    'Speaker_ID': df['Speaker_ID'],
    'L1_Background': df['L1_Background'],
    'Audio_File': df['Sample_File'],
    'Rater_1_Tim': df['Accent_Strength'],
    'Rater_2': '',
    'Rater_3': '',
    'Rater_4': '',
    'Rater_5': ''
})

# Save
output_path = "../accent_rating_sheet_5raters.csv"
rating_sheet.to_csv(output_path, index=False)

print(f"\n✓ Rating sheet saved to: {output_path}")
print(f"\nSheet contains:")
print(f"  - {len(rating_sheet)} speakers")
print(f"  - Ratings in 'Rater_1_Tim' column")
print(f"  - Empty columns for 4 more raters")
print("\n" + "="*60)
print("Preview:")
print("="*60)
print(rating_sheet.head(10))
print("\n✓ Rating sheet ready.")