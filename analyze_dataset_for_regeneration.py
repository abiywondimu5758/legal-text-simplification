# analyze_dataset_for_regeneration.py
import json
import re
from transformers import AutoTokenizer
from difflib import SequenceMatcher

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("masakhane/afri-byt5-base")

# Load dataset
with open('Dataset/final_dataset/final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def count_sentences(text):
    """Count sentence endings"""
    return text.count("።") + text.count("፤")

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def has_deletion(legal, simplified):
    """Check if simplified version has deletions (not just rewrites)"""
    legal_words = set(re.findall(r'[\u1200-\u137F]+', legal))
    simplified_words = set(re.findall(r'[\u1200-\u137F]+', simplified))
    deleted_words = legal_words - simplified_words
    return len(deleted_words) > 3  # At least 3 words deleted

def is_legal_sounding(text):
    """Check if text still sounds legal (has formal markers)"""
    legal_markers = ["ይገደዳል", "ተፈጻሚነት", "እንደተጠበቀ", "በማንኛውም"]
    return any(marker in text for marker in legal_markers)

# Analyze each pair
needs_regeneration = {
    "weak_simplification": [],  # <30% reduction
    "no_splitting": [],  # Long sentences not split
    "no_deletion": [],  # No content deleted
    "still_legal_sounding": [],  # Simplified still sounds legal
    "high_similarity": []  # >90% similar
}

for i, item in enumerate(data):
    legal = item["legal_sentence"]
    simplified = item["simplified_sentence"]
    
    # Tokenize
    legal_tokens = len(tokenizer(legal)["input_ids"])
    simplified_tokens = len(tokenizer(simplified)["input_ids"])
    ratio = simplified_tokens / legal_tokens if legal_tokens > 0 else 1.0
    
    # Check criteria
    issues = []
    
    # 1. Weak simplification (<30% reduction)
    if ratio > 0.7:
        issues.append("weak_simplification")
        needs_regeneration["weak_simplification"].append({
            "index": i,
            "legal": legal,
            "simplified": simplified,
            "ratio": ratio
        })
    
    # 2. No sentence splitting (long legal, same # sentences)
    legal_sent_count = count_sentences(legal)
    simplified_sent_count = count_sentences(simplified)
    if legal_tokens > 200 and legal_sent_count >= simplified_sent_count:
        issues.append("no_splitting")
        needs_regeneration["no_splitting"].append({
            "index": i,
            "legal": legal,
            "simplified": simplified,
            "legal_sentences": legal_sent_count,
            "simplified_sentences": simplified_sent_count
        })
    
    # 3. No deletion
    if not has_deletion(legal, simplified) and legal_tokens > 150:
        issues.append("no_deletion")
        needs_regeneration["no_deletion"].append({
            "index": i,
            "legal": legal,
            "simplified": simplified
        })
    
    # 4. Still legal-sounding
    if is_legal_sounding(simplified):
        issues.append("still_legal_sounding")
        needs_regeneration["still_legal_sounding"].append({
            "index": i,
            "legal": legal,
            "simplified": simplified
        })
    
    # 5. High similarity
    sim = similarity(legal, simplified)
    if sim > 0.9:
        issues.append("high_similarity")
        needs_regeneration["high_similarity"].append({
            "index": i,
            "legal": legal,
            "simplified": simplified,
            "similarity": sim
        })

# Print summary
print("=" * 80)
print("DATASET REGENERATION ANALYSIS")
print("=" * 80)
print(f"\nTotal pairs: {len(data)}")
print(f"\nPairs needing regeneration:")
print(f"  Weak simplification (<30% reduction): {len(needs_regeneration['weak_simplification'])}")
print(f"  No sentence splitting: {len(needs_regeneration['no_splitting'])}")
print(f"  No deletion: {len(needs_regeneration['no_deletion'])}")
print(f"  Still legal-sounding: {len(needs_regeneration['still_legal_sounding'])}")
print(f"  High similarity (>90%): {len(needs_regeneration['high_similarity'])}")

# Get unique indices that need regeneration
all_indices = set()
for category in needs_regeneration.values():
    for item in category:
        all_indices.add(item["index"])

print(f"\n📊 Total unique pairs needing regeneration: {len(all_indices)}")
print(f"   (This is {100*len(all_indices)/len(data):.1f}% of dataset)")

# Save candidates for regeneration
regeneration_candidates = [data[i] for i in sorted(all_indices)]
with open('Dataset/final_dataset/regeneration_candidates.json', 'w', encoding='utf-8') as f:
    json.dump(regeneration_candidates, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(regeneration_candidates)} candidates to:")
print("   Dataset/final_dataset/regeneration_candidates.json")