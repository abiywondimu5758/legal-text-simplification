# format_for_gemini_batch.py
import json

with open('Dataset/final_dataset/top_800_priority.json', 'r', encoding='utf-8') as f:
    pairs = json.load(f)

# Create batches of 50 for easier processing
batch_size = 50
batches = []

for i in range(0, len(pairs), batch_size):
    batch = pairs[i:i+batch_size]
    batches.append({
        "batch_number": i // batch_size + 1,
        "pairs": batch
    })

# Save batches
for batch in batches:
    filename = f'Dataset/final_dataset/gemini_batch_{batch["batch_number"]:02d}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(batch["pairs"], f, ensure_ascii=False, indent=2)

print(f"Created {len(batches)} batches of ~50 pairs each")
print(f"   Files: Dataset/final_dataset/gemini_batch_01.json through gemini_batch_{len(batches):02d}.json")