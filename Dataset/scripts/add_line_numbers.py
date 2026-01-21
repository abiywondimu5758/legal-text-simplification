import os

IN_FILE = "dataset/extracted_sentences/difference.txt"

# Get the absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
file_path = os.path.join(project_root, IN_FILE)

# Read the file
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Add line numbers to each line
numbered_lines = []
for i, line in enumerate(lines, start=201):
    # Prepend line number to each line
    if line.endswith('\n'):
        # Line with newline
        numbered_lines.append(f"{i}. {line}")
    else:
        # Last line without newline
        numbered_lines.append(f"{i}. {line}\n")

# Write back to the file
with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(numbered_lines)

print(f"Added line numbers to {len(lines)} lines in {IN_FILE}")

