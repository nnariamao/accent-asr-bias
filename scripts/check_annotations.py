import os

L2_ARCTIC_PATH = "/Users/timnariamao/accent_bias_project/data/l2arctic_release_v5"
sample_speaker = "ABA"

annotation_path = os.path.join(L2_ARCTIC_PATH, sample_speaker, "annotation")
annotation_files = os.listdir(annotation_path)

print(f"Annotation folder has {len(annotation_files)} files")
print(f"First file: {annotation_files[0]}")

sample_file = os.path.join(annotation_path, annotation_files[0])

print("\n" + "="*50)
print("Content of sample annotation file:")
print("="*50)

with open(sample_file, 'r', encoding='utf-8') as f:
    content = f.read()
    print(content[:1000])