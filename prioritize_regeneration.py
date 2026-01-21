# prioritize_regeneration.py
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("masakhane/afri-byt5-base")

with open('Dataset/final_dataset/regeneration_candidates.json', 'r', encoding='utf-8') as f:
    candidates = json.load(f)

# Score each pair by priority (higher = more urgent)
prioritized = []
for pair in candidates:
    legal = pair["legal_sentence"]
    simplified = pair["simplified_sentence"]
    
    legal_tokens = len(tokenizer(legal)["input_ids"])
    simplified_tokens = len(tokenizer(simplified)["input_ids"])
    ratio = simplified_tokens / legal_tokens if legal_tokens > 0 else 1.0
    
    # Priority score
    priority = 0
    
    # High priority: very weak simplification
    if ratio > 0.85:
        priority += 10
    elif ratio > 0.75:
        priority += 5
    
    # High priority: long sentences not split
    legal_sents = legal.count("።") + legal.count("፤")
    simplified_sents = simplified.count("።") + simplified.count("፤")
    if legal_tokens > 200 and simplified_sents <= legal_sents:
        priority += 8
    
    # Medium priority: still legal-sounding
    if "ይገደዳል" in simplified or "ተፈጻሚነት" in simplified:
        priority += 3
    
    prioritized.append({
        "priority": priority,
        "legal": legal,
        "simplified": simplified,
        "ratio": ratio,
        "legal_tokens": legal_tokens
    })

# Sort by priority (highest first)
prioritized.sort(key=lambda x: x["priority"], reverse=True)

# Save top 800 for regeneration
top_800 = prioritized[:800]
with open('Dataset/final_dataset/top_800_priority.json', 'w', encoding='utf-8') as f:
    json.dump(top_800, f, ensure_ascii=False, indent=2)

print(f"✅ Saved top 800 priority pairs to regenerate")
print(f"   Priority range: {top_800[-1]['priority']} - {top_800[0]['priority']}")