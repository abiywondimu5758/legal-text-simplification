"""
Comprehensive Dataset Analysis for Amharic Legal Text Simplification

This script analyzes the dataset to check:
1. Length distributions and truncation risks
2. Vocabulary diversity
3. Simplification patterns
4. Data quality (duplicates, identical pairs)
5. Learning signal strength
6. Recommendations for improvement

Requirements:
    pip install transformers numpy

The tokenizer will be downloaded to models/afribyt5-base/ on first run.
"""

import json
import os
import numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from difflib import SequenceMatcher
import re

# Create models directory if it doesn't exist
MODEL_DIR = "models/afri-byt5-base"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE DATASET ANALYSIS FOR AMHARIC LEGAL TEXT SIMPLIFICATION")
print("=" * 80)
print("\n[1/8] Loading dataset...")

# Load dataset
with open('Dataset/final_dataset/final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} sentence pairs\n")

print("[2/8] Loading tokenizer (downloading to models/ if needed)...")
# Load tokenizer (will download to local cache, then we can copy to models/)
tokenizer = AutoTokenizer.from_pretrained(
    "masakhane/afri-byt5-base",
    cache_dir=MODEL_DIR
)
print("✓ Tokenizer loaded\n")

# ============================================================================
# 1. LENGTH ANALYSIS
# ============================================================================
print("=" * 80)
print("SECTION 1: LENGTH ANALYSIS")
print("=" * 80)

simplified_lengths = []
legal_lengths = []
length_ratios = []

for item in data:
    legal = item["legal_sentence"]
    simplified = item["simplified_sentence"]
    
    legal_tokens = len(tokenizer(legal)["input_ids"])
    simplified_tokens = len(tokenizer(simplified)["input_ids"])
    
    legal_lengths.append(legal_tokens)
    simplified_lengths.append(simplified_tokens)
    length_ratios.append(simplified_tokens / legal_tokens if legal_tokens > 0 else 0)

print(f"\n--- Legal Sentences (Input) ---")
print(f"Mean: {np.mean(legal_lengths):.1f} tokens")
print(f"Median: {np.median(legal_lengths):.1f} tokens")
print(f"Min: {np.min(legal_lengths)} tokens")
print(f"Max: {np.max(legal_lengths)} tokens")
print(f"Std dev: {np.std(legal_lengths):.1f} tokens")

print(f"\n--- Simplified Sentences (Target) ---")
print(f"Mean: {np.mean(simplified_lengths):.1f} tokens")
print(f"Median: {np.median(simplified_lengths):.1f} tokens")
print(f"Min: {np.min(simplified_lengths)} tokens")
print(f"Max: {np.max(simplified_lengths)} tokens")
print(f"Std dev: {np.std(simplified_lengths):.1f} tokens")

print(f"\n--- Simplification Ratio (Simplified/Legal) ---")
print(f"Mean ratio: {np.mean(length_ratios):.3f} ({100*np.mean(length_ratios):.1f}% of original)")
print(f"Median ratio: {np.median(length_ratios):.3f}")
print(f"Min ratio: {np.min(length_ratios):.3f}")
print(f"Max ratio: {np.max(length_ratios):.3f}")

# Truncation analysis
exceeds_384 = sum(1 for l in simplified_lengths if l > 384)
exceeds_256 = sum(1 for l in simplified_lengths if l > 256)
exceeds_512 = sum(1 for l in legal_lengths if l > 512)

print(f"\n--- Truncation Risk Analysis ---")
print(f"Simplified sentences > 256 tokens: {exceeds_256} ({100*exceeds_256/len(data):.1f}%)")
print(f"Simplified sentences > 384 tokens: {exceeds_384} ({100*exceeds_384/len(data):.1f}%)")
print(f"Legal sentences > 512 tokens: {exceeds_512} ({100*exceeds_512/len(data):.1f}%)")

if exceeds_384 > 0:
    print(f"\n⚠️  WARNING: {exceeds_384} simplified sentences exceed 384 tokens!")
    print("   Consider increasing max_output_length or filtering these samples.")

# Percentiles
print(f"\n--- Percentiles (Simplified) ---")
for p in [50, 75, 90, 95, 99]:
    val = np.percentile(simplified_lengths, p)
    print(f"{p}th percentile: {val:.1f} tokens")

# ============================================================================
# 2. VOCABULARY DIVERSITY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: VOCABULARY DIVERSITY")
print("=" * 80)

all_legal_words = []
all_simplified_words = []
legal_vocab = set()
simplified_vocab = set()

# Simple word tokenization (split by spaces and punctuation)
def tokenize_amharic(text):
    # Split by spaces and common punctuation
    words = re.findall(r'[\u1200-\u137F]+', text)  # Amharic characters
    return [w for w in words if len(w) > 0]

for item in data:
    legal_words = tokenize_amharic(item["legal_sentence"])
    simplified_words = tokenize_amharic(item["simplified_sentence"])
    
    all_legal_words.extend(legal_words)
    all_simplified_words.extend(simplified_words)
    legal_vocab.update(legal_words)
    simplified_vocab.update(simplified_words)

print(f"\n--- Vocabulary Size ---")
print(f"Unique words in legal sentences: {len(legal_vocab)}")
print(f"Unique words in simplified sentences: {len(simplified_vocab)}")
print(f"Total unique words: {len(legal_vocab | simplified_vocab)}")
print(f"Words only in legal: {len(legal_vocab - simplified_vocab)}")
print(f"Words only in simplified: {len(simplified_vocab - legal_vocab)}")
print(f"Common words: {len(legal_vocab & simplified_vocab)}")

# Most common words
legal_word_freq = Counter(all_legal_words)
simplified_word_freq = Counter(all_simplified_words)

print(f"\n--- Most Common Words (Legal) ---")
for word, count in legal_word_freq.most_common(10):
    print(f"  {word}: {count} ({100*count/len(all_legal_words):.2f}%)")

print(f"\n--- Most Common Words (Simplified) ---")
for word, count in simplified_word_freq.most_common(10):
    print(f"  {word}: {count} ({100*count/len(all_simplified_words):.2f}%)")

# Vocabulary richness (type-token ratio)
legal_ttr = len(legal_vocab) / len(all_legal_words) if all_legal_words else 0
simplified_ttr = len(simplified_vocab) / len(all_simplified_words) if all_simplified_words else 0

print(f"\n--- Vocabulary Richness (Type-Token Ratio) ---")
print(f"Legal: {legal_ttr:.4f} (higher = more diverse)")
print(f"Simplified: {simplified_ttr:.4f}")

# ============================================================================
# 3. SIMPLIFICATION PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: SIMPLIFICATION PATTERNS")
print("=" * 80)

# Check for common legal phrases and their simplifications
legal_phrases = [
    "አለበት", "ይገባል", "አይፈቀድም", "ይፈቀዳል",
    "በመሠረት", "እንደ", "ስለ", "ከ...በስተቀር",
    "አንቀጽ", "ሕግ", "ድንጋጌ"
]

phrase_mappings = defaultdict(list)

for item in data:
    legal = item["legal_sentence"]
    simplified = item["simplified_sentence"]
    
    for phrase in legal_phrases:
        if phrase in legal:
            # Check if phrase appears in simplified version
            if phrase in simplified:
                phrase_mappings[phrase].append("preserved")
            else:
                phrase_mappings[phrase].append("removed")

print(f"\n--- Legal Phrase Preservation ---")
for phrase in legal_phrases:
    if phrase_mappings[phrase]:
        preserved = sum(1 for x in phrase_mappings[phrase] if x == "preserved")
        total = len(phrase_mappings[phrase])
        pct = 100 * preserved / total if total > 0 else 0
        print(f"  '{phrase}': {preserved}/{total} preserved ({pct:.1f}%)")

# Check for sentence splitting
split_count = 0
for item in data:
    legal = item["legal_sentence"]
    simplified = item["simplified_sentence"]
    
    # Count sentence endings
    legal_sentences = legal.count("።") + legal.count("፤")
    simplified_sentences = simplified.count("።") + simplified.count("፤")
    
    if simplified_sentences > legal_sentences:
        split_count += 1

print(f"\n--- Sentence Structure Changes ---")
print(f"Sentences that were split: {split_count} ({100*split_count/len(data):.1f}%)")

# ============================================================================
# 4. DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: DATA QUALITY CHECKS")
print("=" * 80)

# Check for duplicates
legal_texts = [item["legal_sentence"] for item in data]
simplified_texts = [item["simplified_sentence"] for item in data]

duplicate_legal = len(legal_texts) - len(set(legal_texts))
duplicate_simplified = len(simplified_texts) - len(set(simplified_texts))

print(f"\n--- Duplicates ---")
print(f"Duplicate legal sentences: {duplicate_legal}")
print(f"Duplicate simplified sentences: {duplicate_simplified}")

# Check for near-duplicates (high similarity)
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

high_similarity_pairs = []
for i, item1 in enumerate(data):
    for j, item2 in enumerate(data[i+1:], i+1):
        sim = similarity(item1["legal_sentence"], item2["legal_sentence"])
        if sim > 0.9:  # 90% similar
            high_similarity_pairs.append((i, j, sim))

print(f"\n--- Near-Duplicates (Legal >90% similar) ---")
print(f"Pairs with >90% similarity: {len(high_similarity_pairs)}")
if len(high_similarity_pairs) > 0:
    print("  Sample pairs:")
    for idx, (i, j, sim) in enumerate(high_similarity_pairs[:3]):
        print(f"    Pair {idx+1} (similarity: {sim:.2f}):")
        print(f"      Legal {i}: {data[i]['legal_sentence'][:80]}...")
        print(f"      Legal {j}: {data[j]['legal_sentence'][:80]}...")

# Check for identical pairs (no simplification)
identical_pairs = []
for i, item in enumerate(data):
    if item["legal_sentence"].strip() == item["simplified_sentence"].strip():
        identical_pairs.append(i)

print(f"\n--- Identical Pairs (No Simplification) ---")
print(f"Pairs where legal == simplified: {len(identical_pairs)}")
if len(identical_pairs) > 0:
    print("  ⚠️  WARNING: These pairs provide no learning signal!")
    for idx in identical_pairs[:5]:
        print(f"    Index {idx}: {data[idx]['legal_sentence'][:100]}...")

# Check for very similar pairs (minimal simplification)
minimal_simplification = []
for i, item in enumerate(data):
    sim = similarity(item["legal_sentence"], item["simplified_sentence"])
    if sim > 0.95:  # 95% similar
        minimal_simplification.append((i, sim))

print(f"\n--- Minimal Simplification (>95% similar) ---")
print(f"Pairs with >95% similarity: {len(minimal_simplification)}")
if len(minimal_simplification) > 0:
    print("  ⚠️  These may not provide strong learning signal")

# Check for empty or very short sentences
empty_legal = [i for i, item in enumerate(data) if len(item["legal_sentence"].strip()) < 10]
empty_simplified = [i for i, item in enumerate(data) if len(item["simplified_sentence"].strip()) < 10]

print(f"\n--- Empty/Too Short Sentences ---")
print(f"Legal sentences <10 chars: {len(empty_legal)}")
print(f"Simplified sentences <10 chars: {len(empty_simplified)}")

# ============================================================================
# 5. SIMPLIFICATION CONSISTENCY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: SIMPLIFICATION CONSISTENCY")
print("=" * 80)

# Check if same legal phrases are simplified consistently
phrase_simplifications = defaultdict(list)

for item in data:
    legal = item["legal_sentence"]
    simplified = item["simplified_sentence"]
    
    # Look for common legal patterns
    if "አለበት" in legal:
        # Extract context around this word
        idx = legal.find("አለበት")
        context = legal[max(0, idx-20):min(len(legal), idx+20)]
        phrase_simplifications["አለበት"].append((context, simplified))

# Check consistency for common patterns
print(f"\n--- Pattern Consistency Check ---")
for pattern, examples in list(phrase_simplifications.items())[:3]:
    print(f"\nPattern: '{pattern}' ({len(examples)} occurrences)")
    # Show a few examples
    for i, (context, simplified) in enumerate(examples[:3]):
        print(f"  Example {i+1}:")
        print(f"    Context: ...{context}...")
        print(f"    Simplified: {simplified[:100]}...")

# ============================================================================
# 6. LENGTH DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: LENGTH DISTRIBUTION ANALYSIS")
print("=" * 80)

# Categorize by length
short_legal = sum(1 for l in legal_lengths if l < 50)
medium_legal = sum(1 for l in legal_lengths if 50 <= l < 150)
long_legal = sum(1 for l in legal_lengths if 150 <= l < 300)
very_long_legal = sum(1 for l in legal_lengths if l >= 300)

print(f"\n--- Legal Sentence Length Distribution ---")
print(f"Short (<50 tokens): {short_legal} ({100*short_legal/len(data):.1f}%)")
print(f"Medium (50-150 tokens): {medium_legal} ({100*medium_legal/len(data):.1f}%)")
print(f"Long (150-300 tokens): {long_legal} ({100*long_legal/len(data):.1f}%)")
print(f"Very Long (≥300 tokens): {very_long_legal} ({100*very_long_legal/len(data):.1f}%)")

short_simplified = sum(1 for l in simplified_lengths if l < 50)
medium_simplified = sum(1 for l in simplified_lengths if 50 <= l < 150)
long_simplified = sum(1 for l in simplified_lengths if 150 <= l < 300)
very_long_simplified = sum(1 for l in simplified_lengths if l >= 300)

print(f"\n--- Simplified Sentence Length Distribution ---")
print(f"Short (<50 tokens): {short_simplified} ({100*short_simplified/len(data):.1f}%)")
print(f"Medium (50-150 tokens): {medium_simplified} ({100*medium_simplified/len(data):.1f}%)")
print(f"Long (150-300 tokens): {long_simplified} ({100*long_simplified/len(data):.1f}%)")
print(f"Very Long (≥300 tokens): {very_long_simplified} ({100*very_long_simplified/len(data):.1f}%)")

# ============================================================================
# 7. LEARNING SIGNAL STRENGTH
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: LEARNING SIGNAL STRENGTH")
print("=" * 80)

# Calculate how much each pair teaches the model
strong_signals = 0  # Significant simplification (>30% reduction)
moderate_signals = 0  # Moderate simplification (10-30% reduction)
weak_signals = 0  # Minimal simplification (<10% reduction)
negative_signals = 0  # Simplified is longer (shouldn't happen)

for i, item in enumerate(data):
    ratio = length_ratios[i]
    if ratio < 0.7:  # >30% reduction
        strong_signals += 1
    elif ratio < 0.9:  # 10-30% reduction
        moderate_signals += 1
    elif ratio <= 1.0:  # <10% reduction
        weak_signals += 1
    else:  # Simplified is longer
        negative_signals += 1

print(f"\n--- Simplification Strength Distribution ---")
print(f"Strong signals (>30% reduction): {strong_signals} ({100*strong_signals/len(data):.1f}%)")
print(f"Moderate signals (10-30% reduction): {moderate_signals} ({100*moderate_signals/len(data):.1f}%)")
print(f"Weak signals (<10% reduction): {weak_signals} ({100*weak_signals/len(data):.1f}%)")
if negative_signals > 0:
    print(f"⚠️  Negative signals (simplified longer): {negative_signals} ({100*negative_signals/len(data):.1f}%)")

# ============================================================================
# 8. RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: RECOMMENDATIONS")
print("=" * 80)

recommendations = []

if exceeds_384 > len(data) * 0.1:
    recommendations.append("⚠️  Consider increasing max_output_length to 512 or filtering long samples")

if duplicate_legal > 0:
    recommendations.append("⚠️  Remove duplicate legal sentences to improve diversity")

if len(identical_pairs) > 0:
    recommendations.append("⚠️  Remove or fix identical pairs (no learning signal)")

if len(minimal_simplification) > len(data) * 0.2:
    recommendations.append("⚠️  Many pairs have minimal simplification - consider more aggressive simplification")

if weak_signals > len(data) * 0.3:
    recommendations.append("⚠️  High proportion of weak signals - dataset may not teach strong simplification patterns")

if legal_ttr < 0.1:
    recommendations.append("⚠️  Low vocabulary diversity in legal sentences - consider adding more varied legal text")

if len(high_similarity_pairs) > len(data) * 0.1:
    recommendations.append("⚠️  Many near-duplicate pairs - consider removing to improve diversity")

if not recommendations:
    recommendations.append("✓ Dataset looks good! No major issues detected.")

print("\n--- Summary of Recommendations ---")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nTotal pairs analyzed: {len(data)}")
print(f"Dataset size: {len(data)} training examples")
print(f"Estimated trainable patterns: ~{strong_signals + moderate_signals} strong-to-moderate simplification examples")
