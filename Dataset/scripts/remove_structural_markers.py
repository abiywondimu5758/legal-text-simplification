import re

IN_FILE = "dataset/extracted_sentences/candidate_complex_sentences_normalized_sentences.txt"
OUT_FILE = "dataset/extracted_sentences/candidate_complex_sentences_model.txt"


def clean_structure(sentence):
    # Remove standalone article headers ONLY at sentence start
    # Handles clean + OCR-corrupted forms:
    # አንቀጽ 4
    # አንቀጽ ፩፻፱
    # አንቀጽ 406.
    # አንቀጽ 4()5.
    sentence = re.sub(
        r"^\s*አንቀጽ\s+[^\s።፤]+",
        "",
        sentence
    )

    # Remove chapter / section headers ONLY at sentence start
    sentence = re.sub(r"^\s*ምዕራፍ\s+\S+", "", sentence)
    sentence = re.sub(r"^\s*ክፍል\s+\S+", "", sentence)

    # Remove clause numbering (Ethiopic numerals) like ፩/, ፪/
    sentence = re.sub(r"\b[፩-፺]+\/", "", sentence)

    # Remove clause numbering (Arabic numerals) like 1/, 2 /, 3/
    sentence = re.sub(r"\b\d+\s*\/", "", sentence)

    # Normalize extra spaces
    sentence = re.sub(r"\s{2,}", " ", sentence)

    return sentence.strip()

with open(IN_FILE, "r", encoding="utf-8") as f:
    sentences = f.readlines()

cleaned = [clean_structure(s) for s in sentences if s.strip()]

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in cleaned:
        f.write(s + "\n")

print(f"Cleaned sentences written: {len(cleaned)}")

