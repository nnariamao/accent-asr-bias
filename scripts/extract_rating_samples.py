import os
import pandas as pd
import shutil
import random

print("="*60)
print("Extracting Audio Samples for Accent Rating")
print("="*60)

# Load speaker inventory
inventory = pd.read_csv("/Users/timnariamao/accent_bias_project/speaker_inventory.csv")
L2_ARCTIC_PATH = "/Users/timnariamao/accent_bias_project/data/l2arctic_release_v5"

# Create output directory
samples_dir = "/Users/timnariamao/accent_bias_project/rating_samples"
os.makedirs(samples_dir, exist_ok=True)

# Extract one random file from each speaker
sample_info = []

for idx, row in inventory.iterrows():
    speaker_id = row['Speaker_ID']
    l1_background = row['L1_Background']
    
    wav_dir = os.path.join(L2_ARCTIC_PATH, speaker_id, "wav")
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    # Select a random file
    random.seed(42)  # For reproducibility
    selected_file = random.choice(wav_files)
    
    # Copy to rating_samples folder with speaker ID prefix
    source = os.path.join(wav_dir, selected_file)
    dest = os.path.join(samples_dir, f"{speaker_id}_{selected_file}")
    shutil.copy(source, dest)
    
    sample_info.append({
        'Speaker_ID': speaker_id,
        'L1_Background': l1_background,
        'Sample_File': f"{speaker_id}_{selected_file}",
        'Original_File': selected_file
    })
    
    print(f"✓ {speaker_id} ({l1_background}): {selected_file}")

# Save sample info
sample_df = pd.DataFrame(sample_info)
sample_df.to_csv("/Users/timnariamao/accent_bias_project/rating_samples_info.csv", index=False)

