import fitz  # pymupdf
import os

RAW_DIR = "dataset/raw_legal_text"
OUT_DIR = "dataset/extracted_sentences"
os.makedirs(OUT_DIR, exist_ok=True)

output_path = os.path.join(OUT_DIR, "raw_extracted_text.txt")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        t = page.get_text("text")
        if t:
            text.append(t)
    return "\n".join(text)

with open(output_path, "w", encoding="utf-8") as out:
    for fname in os.listdir(RAW_DIR):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, fname)
            print(f"Extracting: {fname}")
            text = extract_text_from_pdf(pdf_path)
            out.write(text)
            out.write("\n\n")

print("PDF extraction finished.")
