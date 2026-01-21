import re

IN_FILE = "dataset/extracted_sentences/all_sentences.txt"
OUT_FILE = "dataset/extracted_sentences/candidate_complex_sentences.txt"

KEYWORDS = [
    "ይገደዳል",
    "አይፈቀድም",
    "ይፈቀዳል",
    "ቢሆንም",
    "ከ",
    "በስተቀር",
    "እንደተጠበቀ",
    "መሠረት",
    "ካል",
    "እስከ"
]

def is_complex(sentence):
    words = sentence.split()
    if len(words) > 15:
        return True
    for kw in KEYWORDS:
        if kw in sentence:
            return True
    return False

kept = []

with open(IN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        if is_complex(s):
            kept.append(s)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in kept:
        f.write(s + "\n")

print(f"Filtered complex sentences: {len(kept)}")
