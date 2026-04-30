import whisper
import os
from jiwer import wer
import pandas as pd
from tqdm import tqdm

print("="*60)
print("Running ASR Test on 3 Speakers")
print("="*60)

# Load Whisper model
print("\nLoading Whisper model...")
model = whisper.load_model("base")
print("✓ Model loaded!")

inventory = pd.read_csv("/Users/timnariamao/accent_bias_project/speaker_inventory.csv")

# Select first 3 speakers for a quick pilot test
test_speakers = inventory.head(3)

print(f"\n✓ Testing with {len(test_speakers)} speakers:")
for idx, row in test_speakers.iterrows():
    print(f"   - {row['Speaker_ID']} (L1: {row['L1_Background']})")

L2_ARCTIC_PATH = "/Users/timnariamao/accent_bias_project/data/l2arctic_release_v5"

all_results = []

# Process each speaker
for idx, row in test_speakers.iterrows():
    speaker_id = row['Speaker_ID']
    l1_background = row['L1_Background']
    
    print(f"\n{'='*60}")
    print(f"Processing Speaker {idx+1}/{len(test_speakers)}: {speaker_id} (L1: {l1_background})")
    print(f"{'='*60}")
    
    wav_dir = os.path.join(L2_ARCTIC_PATH, speaker_id, "wav")
    transcript_dir = os.path.join(L2_ARCTIC_PATH, speaker_id, "transcript")
    
    # Get all wav files
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    
    speaker_wers = []
    
    # Process files with progress bar
    for wav_file in tqdm(wav_files, desc=f"  {speaker_id}", leave=False):
        audio_path = os.path.join(wav_dir, wav_file)
        transcript_file = wav_file.replace('.wav', '.txt')
        transcript_path = os.path.join(transcript_dir, transcript_file)
        
        # Load reference
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                reference = f.read().strip()
        except:
            continue  # Skip if transcript missing
        
        # Transcribe with Whisper
        try:
            result = model.transcribe(audio_path, language="en")
            hypothesis = result["text"].strip()
            
            # Calculate WER
            error_rate = wer(reference, hypothesis)
            speaker_wers.append(error_rate)
            
            # Store detailed results
            all_results.append({
                'speaker_id': speaker_id,
                'l1_background': l1_background,
                'file': wav_file,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': error_rate
            })
        except Exception as e:
            print(f"    Error processing {wav_file}: {e}")
            continue
    
    # Speaker summary
    avg_wer = sum(speaker_wers) / len(speaker_wers) if speaker_wers else 0
    print(f"  ✓ Files processed: {len(speaker_wers)}")
    print(f"  ✓ Average WER: {avg_wer:.3f} ({avg_wer*100:.1f}%)")

# Save all results
print("\n" + "="*60)
print("Saving Results")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df.to_csv("../test_3speakers_results.csv", index=False)
print(f"✓ Detailed results saved to: ../test_3speakers_results.csv")

# Create speaker-level summary
speaker_summary = results_df.groupby(['speaker_id', 'l1_background'])['wer'].agg(['mean', 'std', 'count']).reset_index()
speaker_summary.columns = ['Speaker_ID', 'L1_Background', 'Mean_WER', 'Std_WER', 'Num_Files']
speaker_summary = speaker_summary.sort_values('Mean_WER', ascending=False)
speaker_summary.to_csv("../test_3speakers_summary.csv", index=False)
print(f"✓ Speaker summary saved to: ../test_3speakers_summary.csv")

print("\n" + "="*60)
print("Test Statistics")
print("="*60)
print(f"Total utterances processed: {len(results_df)}")
print(f"Average WER across 3 speakers: {results_df['wer'].mean():.3f} ({results_df['wer'].mean()*100:.1f}%)")
print(f"\nWER by Speaker:")
print(speaker_summary[['Speaker_ID', 'L1_Background', 'Mean_WER']])

print("\n" + "="*60)
print("3-Speaker Test Complete!")
print("="*60)
