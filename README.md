# accent-asr-bias

A study examining whether gradient accent strength or categorical L1 background better predicts ASR word error rate (WER) on L2-accented English speech, using OpenAI Whisper and the L2-ARCTIC corpus.

## Data

The dataset used in this project is the **L2-ARCTIC corpus** (v5), a speech corpus of non-native English speakers across six L1 backgrounds: Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese.

The dataset is not included in this repository due to its size (approx. 8.3 GB). You can download it here:

**https://psi.engr.tamu.edu/l2-arctic-corpus/**

Once downloaded, place the extracted folder at:

```
accent_bias_project/data/l2arctic_release_v5/
```

## Project Structure

```
scripts/                  Analysis and processing scripts
figures/                  Generated figures (300 dpi PNG)
results/                  Output CSVs from ASR evaluation
accent_rating_sheet_5raters.csv    Raw ratings from all 5 raters
accent_ratings_5raters.csv         Processed ratings (mean, SD per speaker)
speaker_wer_summary.csv            Per-speaker WER from Whisper
merged_analysis_multirater.csv     Merged accent and WER data
stats_results.json                 Machine-readable summary statistics
analysis_summary.txt               Full analysis report
```

## Scripts

Run in the following order:

1. `scripts/explore_data.py` - Explore the L2-ARCTIC dataset structure
2. `scripts/run_full_asr.py` - Run Whisper ASR on all speakers
3. `scripts/extract_rating_samples.py` - Extract audio samples for rating
4. `scripts/process_multi_rater_data.py` - Process completed rating sheet
5. `scripts/final_analysis_multirater.py` - Run statistical analyses
6. `scripts/generate_figures.py` - Generate all figures

## Requirements

```
pip install openai-whisper jiwer pandas numpy scipy scikit-learn matplotlib seaborn tqdm
```
