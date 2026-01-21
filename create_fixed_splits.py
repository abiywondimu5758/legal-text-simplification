"""
Create fixed train/validation/test splits for final.json and regen.json datasets.

This ensures consistency across all training notebooks - Models 1 & 2 will use
the same splits from final.json, and Models 3 & 4 will use the same splits from regen.json.
"""

import json
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
WORKDIR = Path(__file__).parent
DATASET_DIR = WORKDIR / "Dataset" / "final_dataset"

FINAL_JSON = DATASET_DIR / "final.json"
REGEN_JSON = DATASET_DIR / "regen.json"

# Output paths
FINAL_TRAIN = DATASET_DIR / "final_train.json"
FINAL_VAL = DATASET_DIR / "final_val.json"
FINAL_TEST = DATASET_DIR / "final_test.json"

REGEN_TRAIN = DATASET_DIR / "regen_train.json"
REGEN_VAL = DATASET_DIR / "regen_val.json"
REGEN_TEST = DATASET_DIR / "regen_test.json"

# Split sizes
FINAL_TRAIN_SIZE = 1700
FINAL_VAL_SIZE = 200
FINAL_TEST_SIZE = 100  # Remaining will be test

REGEN_TRAIN_SIZE = 1700
REGEN_VAL_SIZE = 200
REGEN_TEST_SIZE = 100  # Remaining will be test


def create_splits(input_file, train_file, val_file, test_file, train_size, val_size, test_size):
    """Create fixed train/val/test splits from input file"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*60}")
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Shuffle with fixed seed
    indices = np.random.permutation(len(data))
    shuffled_data = [data[i] for i in indices]
    
    # Calculate actual split sizes
    total_requested = train_size + val_size + test_size
    if len(data) < total_requested:
        print(f" Warning: Dataset has {len(data)} samples, but requested {total_requested}")
        print(f"   Adjusting split sizes...")
        train_size = int(len(data) * 0.85)
        val_size = int(len(data) * 0.10)
        test_size = len(data) - train_size - val_size
    
    # Split data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:train_size + val_size + test_size]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)}")
    
    # Verify no overlap
    train_sentences = set([item["legal_sentence"] for item in train_data])
    val_sentences = set([item["legal_sentence"] for item in val_data])
    test_sentences = set([item["legal_sentence"] for item in test_data])
    
    train_val_overlap = train_sentences.intersection(val_sentences)
    train_test_overlap = train_sentences.intersection(test_sentences)
    val_test_overlap = val_sentences.intersection(test_sentences)
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"\n WARNING: Found overlaps!")
        if train_val_overlap:
            print(f"  Train-Val overlap: {len(train_val_overlap)} sentences")
        if train_test_overlap:
            print(f"  Train-Test overlap: {len(train_test_overlap)} sentences")
        if val_test_overlap:
            print(f"  Val-Test overlap: {len(val_test_overlap)} sentences")
    else:
        print(f"\n Verified: No overlaps between splits!")
    
    # Save splits
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {train_file}")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {val_file}")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {test_file}")
    
    return train_data, val_data, test_data


def main():
    """Create fixed splits for both datasets"""
    
    print("="*60)
    print("Creating Fixed Dataset Splits")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Working directory: {WORKDIR}")
    
    # Check if input files exist
    if not FINAL_JSON.exists():
        print(f"\nError: {FINAL_JSON} not found!")
        return
    
    if not REGEN_JSON.exists():
        print(f"\n Error: {REGEN_JSON} not found!")
        return
    
    # Create splits for final.json
    print("\n" + "="*60)
    print("Creating splits for final.json")
    print("="*60)
    final_train, final_val, final_test = create_splits(
        FINAL_JSON,
        FINAL_TRAIN,
        FINAL_VAL,
        FINAL_TEST,
        FINAL_TRAIN_SIZE,
        FINAL_VAL_SIZE,
        FINAL_TEST_SIZE
    )
    
    # Create splits for regen.json
    print("\n" + "="*60)
    print("Creating splits for regen.json")
    print("="*60)
    regen_train, regen_val, regen_test = create_splits(
        REGEN_JSON,
        REGEN_TRAIN,
        REGEN_VAL,
        REGEN_TEST,
        REGEN_TRAIN_SIZE,
        REGEN_VAL_SIZE,
        REGEN_TEST_SIZE
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nfinal.json splits:")
    print(f"  Train: {len(final_train)} samples → {FINAL_TRAIN}")
    print(f"  Val:   {len(final_val)} samples → {FINAL_VAL}")
    print(f"  Test:  {len(final_test)} samples → {FINAL_TEST}")
    
    print("\nregen.json splits:")
    print(f"  Train: {len(regen_train)} samples → {REGEN_TRAIN}")
    print(f"  Val:   {len(regen_val)} samples → {REGEN_VAL}")
    print(f"  Test:  {len(regen_test)} samples → {REGEN_TEST}")
    
    print("\nAll splits created successfully!")
    print("\nNext steps:")
    print("  1. Use final_train.json, final_val.json, final_test.json for Models 1 & 2")
    print("  2. Use regen_train.json, regen_val.json, regen_test.json for Models 3 & 4")
    print("  3. Update training notebooks to load these pre-split files")


if __name__ == "__main__":
    main()

