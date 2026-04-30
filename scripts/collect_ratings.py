import pandas as pd
import os

print("="*60)
print("Accent Strength Rating Collection")
print("="*60)

# Load sample info
sample_info = pd.read_csv("/Users/timnariamao/accent_bias_project/rating_samples_info.csv")

print("\nRating Scale:")
print("1 = Minimal accent (near-native)")
print("2 = Slight accent")
print("3 = Moderate accent")
print("4 = Strong accent")
print("5 = Very strong accent")
print("\n" + "="*60)

# Collect ratings
ratings = []

for idx, row in sample_info.iterrows():
    speaker_id = row['Speaker_ID']
    l1 = row['L1_Background']
    audio_file = row['Sample_File']
    
    print(f"\n[{idx+1}/24] Speaker: {speaker_id} (L1: {l1})")
    print(f"Audio file: {audio_file}")
    print("(Listen to the audio, then enter rating)")
    
    while True:
        try:
            rating = int(input("Rating (1-5): "))
            if 1 <= rating <= 5:
                break
            else:
                print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    ratings.append({
        'Speaker_ID': speaker_id,
        'L1_Background': l1,
        'Accent_Strength': rating
    })

# Save ratings
ratings_df = pd.DataFrame(ratings)
ratings_df.to_csv("/Users/timnariamao/accent_bias_project/accent_ratings.csv", index=False)

print("\n" + "="*60)
print("✓ Ratings saved to: accent_ratings.csv")
print("="*60)