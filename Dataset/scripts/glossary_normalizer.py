import csv

INPUT = "Dataset/scripts/glossary/glossary.csv"
OUTPUT = "Dataset/scripts/glossary/glossary_normalized.csv"

EXPECTED_FIELDS = [
    "legal_term",
    "sense_id",
    "plain_term",
    "category",
    "change_allowed",
    "rule_type",
    "context",
    "notes"
]

with open(INPUT, newline="", encoding="utf-8") as infile, \
     open(OUTPUT, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=EXPECTED_FIELDS)
    writer.writeheader()

    for row in reader:
        normalized = {}
        for field in EXPECTED_FIELDS:
            value = row.get(field)
            normalized[field] = (value or "").strip() or "same"
        writer.writerow(normalized)

print("Glossary normalized safely and completely.")

