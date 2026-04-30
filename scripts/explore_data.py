import os
import pandas as pd

# Path to L2-ARCTIC data 
L2_ARCTIC_PATH = "/Users/timnariamao/accent_bias_project/data/l2arctic_release_v5"

print("=" * 50)
print("L2-ARCTIC Dataset Exploration")
print("=" * 50)

# 1. Get all speaker folders
all_items = os.listdir(L2_ARCTIC_PATH)
speakers = [item for item in all_items 
            if os.path.isdir(os.path.join(L2_ARCTIC_PATH, item))
            and item not in ['README', 'PROMPTS', '__MACOSX']]  # Exclude non-speaker items

speakers.sort()  # Alphabetical order
print(f"\n✓ Found {len(speakers)} speakers")
print(f"  Speaker IDs: {speakers}\n")

# 2. Explore one speaker in detail
sample_speaker = speakers[0]
print(f"Exploring sample speaker: {sample_speaker}")
print("-" * 50)

speaker_path = os.path.join(L2_ARCTIC_PATH, sample_speaker)
subfolders = ['annotation', 'textgrid', 'transcript', 'wav']

for subfolder in subfolders:
    folder_path = os.path.join(speaker_path, subfolder)
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        print(f"{subfolder:12} → {len(files):4} files (samples: {files[:2]})")
    else:
        print(f"{subfolder:12} → NOT FOUND")

# 3. Check transcript format
print("\n" + "=" * 50)
print("Sample Transcript Content")
print("=" * 50)

transcript_folder = os.path.join(speaker_path, "transcript")
transcript_files = [f for f in os.listdir(transcript_folder) if f.endswith('.txt')]

if transcript_files:
    sample_transcript = transcript_files[0]
    transcript_path = os.path.join(transcript_folder, sample_transcript)
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"File: {sample_transcript}")
    print(f"Content: {content}")
else:
    print("No .txt files found in transcript folder")

# 4. Count audio files per speaker
print("\n" + "=" * 50)
print("Audio File Counts Per Speaker")
print("=" * 50)

speaker_audio_counts = {}
for speaker in speakers:
    wav_folder = os.path.join(L2_ARCTIC_PATH, speaker, "wav")
    if os.path.exists(wav_folder):
        wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
        speaker_audio_counts[speaker] = len(wav_files)
    else:
        speaker_audio_counts[speaker] = 0

for speaker, count in speaker_audio_counts.items():
    print(f"{speaker}: {count} audio files")
