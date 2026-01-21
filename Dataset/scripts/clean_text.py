import re

IN_FILE = "dataset/extracted_sentences/raw_extracted_text.txt"
OUT_FILE = "dataset/extracted_sentences/clean_text.txt"

def is_primarily_english(line):
    """Check if a line is primarily English (more than 50% English characters)"""
    if not line.strip():
        return False
    
    # Count English letters (a-z, A-Z)
    english_chars = len(re.findall(r'[a-zA-Z]', line))
    # Count all non-whitespace characters
    total_chars = len(re.findall(r'\S', line))
    
    if total_chars == 0:
        return False
    
    # If more than 50% are English characters, consider it primarily English
    return (english_chars / total_chars) > 0.5

def remove_english_words(line):
    """Remove English words from a line, keeping only non-English content"""
    # Remove English words (sequences of English letters)
    # This regex matches words that contain only a-z, A-Z (and optional numbers)
    line = re.sub(r'\b[a-zA-Z0-9]+\b', '', line)
    # Clean up extra spaces
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # If line is primarily English, skip it entirely (remove paragraph/sentence)
        if is_primarily_english(line):
            continue
        
        # Remove English words from mixed content
        cleaned_line = remove_english_words(line)
        
        # Only keep non-empty lines
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    # Join lines and clean up whitespace
    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)
    return cleaned.strip()

with open(IN_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

# Step 1: Remove English content
cleaned = clean_text(raw)
print("Step 1: Cleaning finished. All English content removed.")

# Step 2: Keep only lines containing Ethiopic characters
ethiopic_re = re.compile(r"[\u1200-\u137F]")
lines = cleaned.split('\n')
clean_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # keep only lines that contain Ethiopic characters
    if ethiopic_re.search(line):
        clean_lines.append(line)
print(f"Step 2: Kept {len(clean_lines)} Amharic lines")

# Step 3: Reconstruct paragraphs
paragraphs = []
buffer = ""
for line in clean_lines:
    line = line.strip()
    if not line:
        continue

    buffer += " " + line

    if "።" in line:
        paragraphs.append(buffer.strip())
        buffer = ""

# flush remaining
if buffer.strip():
    paragraphs.append(buffer.strip())
print(f"Step 3: Reconstructed {len(paragraphs)} paragraphs")

# Step 4: Filter out unwanted patterns
DROP_PATTERNS = [
    "ፌዯራል ነጋሪት ጋዜጣ",
    "የኢትዮጵያ ፌዯራላዊ ዲሞክራሲያዊ ሪፐብሊክ",
    "ሕዝብ ተወካዮች ምክር ቤት",
    "አዋጅ ቁጥር",
    "ማጽዯቂያ አዋጅ",
    "ገጽ",
    "ዓመት ቁጥር",
    "በመሆኑ፤",
    "ይህንኑ ስምምነት",
]

kept = []
for line in paragraphs:
    line = line.strip()
    if not line:
        continue

    drop = False
    for p in DROP_PATTERNS:
        if p in line:
            drop = True
            break

    # also drop extremely long preambles
    if len(line.split()) > 80 and "ይገደዳል" not in line and "አለበት" not in line:
        drop = True

    if not drop:
        kept.append(line)

# Write final result to clean_text.txt
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for l in kept:
        f.write(l + "\n")

print(f"Step 4: Kept {len(kept)} legal content lines")
print("All processing complete. Final output written to clean_text.txt")
