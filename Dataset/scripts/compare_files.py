import os

FILE1 = "dataset/extracted_sentences/candidate_complex_sentences_model.txt"
FILE2 = "dataset/seed_pairs/gold_seed_candidates.txt"
OUT_FILE = "dataset/extracted_sentences/difference.txt"

# Get the absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

file1_path = os.path.join(project_root, FILE1)
file2_path = os.path.join(project_root, FILE2)
out_file_path = os.path.join(project_root, OUT_FILE)

# Read the second file and create a set of lines for fast lookup
# Strip whitespace and newlines for comparison
print(f"Reading {FILE2}...")
with open(file2_path, "r", encoding="utf-8") as f:
    file2_lines = set(line.strip() for line in f if line.strip())

print(f"Found {len(file2_lines)} non-empty lines in {FILE2}")

# Read the first file and find lines not in the second file
print(f"Reading {FILE1}...")
difference_lines = []

with open(file1_path, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()
        if stripped and stripped not in file2_lines:
            difference_lines.append(line.rstrip('\n'))  # Keep original line content but remove trailing newline

print(f"Found {len(difference_lines)} lines in {FILE1} that are not in {FILE2}")

# Write the differences to the output file
print(f"Writing differences to {OUT_FILE}...")
with open(out_file_path, "w", encoding="utf-8") as f:
    for line in difference_lines:
        f.write(line + "\n")

print(f"Done! Wrote {len(difference_lines)} lines to {OUT_FILE}")







