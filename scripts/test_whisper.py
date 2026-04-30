import whisper
import os
from jiwer import wer
import pandas as pd

print("="*60)
print("Testing Whisper ASR Pipeline")
print("="*60)

# Load Whisper model
print("\nLoading Whisper model (this may take a minute)...")
model = whisper.load_model("base")
print("✓ Model loaded successfully!")

# Test with one speaker
L2_ARCTIC_PATH = "/Users/timnariamao/accent_bias_project/data/l2arctic_release_v5"
test_speaker = "ABA"  # Arabic speaker

print(f"\nTesting with speaker: {test_speaker}")
print("-"*60)

# Get first 5 audio files
wav_dir = os.path.join(L2_ARCTIC_PATH, test_speaker, "wav")
transcript_dir = os.path.join(L2_ARCTIC_PATH, test_speaker, "transcript")

wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])[:5]

results = []

for wav_file in wav_files:
    # File paths
    audio_path = os.path.join(wav_dir, wav_file)
    transcript_file = wav_file.replace('.wav', '.txt')
    transcript_path = os.path.join(transcript_dir, transcript_file)
    
    # Load reference transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()
    
    # Run Whisper
    print(f"\nProcessing: {wav_file}")
    result = model.transcribe(audio_path, language = "en")
    hypothesis = result["text"].strip()
    
    # Calculate WER
    error_rate = wer(reference, hypothesis)
    
    # Store results
    results.append({
        'file': wav_file,
        'reference': reference,
        'hypothesis': hypothesis,
        'wer': error_rate
    })
    
    # Print comparison
    print(f"  Reference:  {reference}")
    print(f"  Whisper:    {hypothesis}")
    print(f"  WER:        {error_rate:.3f} ({error_rate*100:.1f}%)")

# Summary
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

df = pd.DataFrame(results)
avg_wer = df['wer'].mean()

print(f"Files processed: {len(results)}")
print(f"Average WER: {avg_wer:.3f} ({avg_wer*100:.1f}%)")
print(f"\nDetailed results saved to: ../test_whisper_results.csv")

df.to_csv("../test_whisper_results.csv", index=False)

print("\n" + "="*60)
print("Pipeline Test Complete!")
print("="*60)