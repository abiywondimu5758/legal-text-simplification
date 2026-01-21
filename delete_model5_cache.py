"""
Delete all cached evaluation results for Model 5
This script deletes all cache files for Model 5 (all 4 combinations: with/without RAG, with/without Contrastive)
"""

from pathlib import Path
import json

# Constants
WORKDIR = Path(__file__).parent
EVALUATION_CACHE_DIR = WORKDIR / "evaluation_cache"

def delete_model_5_cache():
    """Delete all cached evaluation results for Model 5 (all 4 combinations)"""
    deleted_count = 0
    deleted_keys = []
    
    if not EVALUATION_CACHE_DIR.exists():
        print("No cache directory found.")
        return deleted_count, deleted_keys
    
    print(f"Searching for Model 5 cache files in: {EVALUATION_CACHE_DIR}")
    print()
    
    for cache_file in EVALUATION_CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cache_key = data.get('_cache_key', '')
                
                # Check if this is a Model 5 cache file
                if cache_key.startswith("Model 5"):
                    print(f"Found: {cache_key}")
                    cache_file.unlink()  # Delete the file
                    deleted_count += 1
                    deleted_keys.append(cache_key)
        except Exception as e:
            print(f"Error processing {cache_file}: {e}")
            continue
    
    return deleted_count, deleted_keys

if __name__ == "__main__":
    print("=" * 60)
    print("Deleting Model 5 Cache")
    print("=" * 60)
    print()
    
    deleted_count, deleted_keys = delete_model_5_cache()
    
    print()
    print("=" * 60)
    if deleted_count > 0:
        print(f"✅ Successfully deleted {deleted_count} cache file(s) for Model 5")
        print()
        print("Deleted cache keys:")
        for key in deleted_keys:
            print(f"  - {key}")
    else:
        print("ℹ️  No Model 5 cache files found to delete")
    print("=" * 60)

