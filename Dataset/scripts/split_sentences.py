
import re
IN_FILE = "dataset/extracted_sentences/candidate_complex_sentences_normalized.txt"
OUT_FILE = "dataset/extracted_sentences/candidate_complex_sentences_normalized_sentences.txt"
sentences = []

with open(IN_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize spacing around sentence delimiters FIRST
# This ensures cases like "ቃል።ሌላ" become "ቃል። ሌላ"
text = re.sub(r"(።|፤|፡፡)(\S)", r"\1 \2", text)

# Now split while keeping delimiters
parts = re.split(r"(።|፤|፡፡)", text)

current = ""

for part in parts:
    part = part.strip()

    if part in {"።", "፤","፡፡"}:
        sentence = (current + part).strip()
        if len(sentence) >= 10:
            sentences.append(sentence)
        current = ""
    else:
        if part:
            current += part + " "

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in sentences:
        f.write(s + "\n")

print(f"Final sentence count: {len(sentences)}")
