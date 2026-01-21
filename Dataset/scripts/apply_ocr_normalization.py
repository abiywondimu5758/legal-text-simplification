import os
from glossary.ocr_normalization_map import OCR_MAP


INPUT_FILE = "dataset/extracted_sentences/candidate_complex_sentences_normalized1.txt"
OUTPUT_FILE = "dataset/extracted_sentences/candidate_complex_sentences_normalized2.txt"


def apply_ocr_map(text: str, ocr_map: dict) -> str:
    """
    Apply OCR normalization map to a single string.
    Longer keys are applied first to avoid partial replacement issues.
    """
    for src in sorted(ocr_map.keys(), key=len, reverse=True):
        text = text.replace(src, ocr_map[src])
    return text


def normalize_file(input_path: str, output_path: str, ocr_map: dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write("\n")
                continue

            normalized = apply_ocr_map(line, ocr_map)
            fout.write(normalized + "\n")


if __name__ == "__main__":
    normalize_file(INPUT_FILE, OUTPUT_FILE, OCR_MAP)
    print(f"OCR normalization completed.\nOutput written to: {OUTPUT_FILE}")
