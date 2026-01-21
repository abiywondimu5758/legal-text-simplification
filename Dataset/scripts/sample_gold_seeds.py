import random

IN_FILE = "dataset/extracted_sentences/candidate_complex_sentences_model.txt"
OUT_FILE = "dataset/seed_pairs/gold_seed_candidates.txt"

with open(IN_FILE, "r", encoding="utf-8") as f:
    sentences = [s.strip() for s in f if s.strip()]

def word_count(s):
    return len(s.split())

short = [s for s in sentences if 10 <= word_count(s) <= 15]
medium = [s for s in sentences if 16 <= word_count(s) <= 25]
long = [s for s in sentences if word_count(s) >= 26]

random.seed(42)

selected = (
    random.sample(short, min(50, len(short))) +
    random.sample(medium, min(80, len(medium))) +
    random.sample(long, min(70, len(long)))
)

random.shuffle(selected)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in selected:
        f.write(s + "\n")

print(f"Gold seed candidates written: {len(selected)}")
