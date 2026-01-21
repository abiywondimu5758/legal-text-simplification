# regenerate_parallel.py
import json
import time
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import queue
import threading
import signal
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your 9 API keys
API_KEYS = [
    "REPLACE WITH YOUR API KEY HERE"  # Key 1 - Replace with your 9th key
]

# Model priority list (best to worst) - will fallback automatically on rate limit
MODEL_PRIORITY = [
    "gemini-2.5-flash",           # Best: Stable, June 2025
    "gemini-2.5-pro",             # Fallback 1: Pro version
    "gemini-2.0-flash-001",       # Fallback 2: Stable, January 2025
    "gemini-2.0-flash",           # Fallback 3: Stable
    "gemini-flash-latest",        # Fallback 4: Latest stable
    "gemini-2.5-flash-lite",      # Fallback 5: Lighter version
    "gemini-2.0-flash-lite-001",  # Fallback 6: Lighter stable
    "gemini-3-flash-preview",     # Fallback 7: Preview
    "gemma-3-27b-it",             # Fallback 8: Gemma model
]

# Wait 8 seconds after each response (to stay under rate limits)
POST_REQUEST_WAIT = 8.0

# ============================================================================
# PROMPT TEMPLATE (SAME FOR ALL KEYS - CRITICAL FOR CONSISTENCY)
# ============================================================================

PROMPT_TEMPLATE = """You are an expert at simplifying Amharic legal text for non-legal readers.

TASK:
Rewrite the following legal sentence to make it understandable for ordinary citizens,
while preserving ALL legal meaning.

LEGAL SENTENCE:
{legal_sentence}

REQUIREMENTS (MUST FOLLOW):

1. AGGRESSIVE SIMPLIFICATION REQUIRED:
   - You MUST simplify, not copy
   - Prefer two or more short sentences over one long sentence
   - Delete redundant legal boilerplate
   - Use everyday Amharic, not legal-code language

2. SENTENCE SPLITTING:
   - If the sentence is long OR contains multiple clauses, SPLIT it into 2–3 shorter sentences
   - Each sentence must be grammatically complete and clear

3. DELETION OF BOILERPLATE:
   - Remove phrases that add no legal meaning, such as:
     * "እንደተጠበቀ ሆኖ"
     * "በማንኛውም ሁኔታ"
     * "ያለ አግባብ"
   - Do NOT delete conditions, obligations, or legal effects

4. VOCABULARY SIMPLIFICATION:
   - Replace archaic legal wording with common Amharic:
     * "ይገደዳል" → "አለበት"
     * "ተፈጻሚነት" → "መፈጸም" or "መስራት"
   - Prefer commonly used government or newspaper language

5. STRUCTURE SIMPLIFICATION:
   - Break nested clauses into linear order
   - Make passive constructions active when possible
   - Make implicit actors explicit when clear

6. DEFINITION OR EXPANSION (WHEN NECESSARY):
   - If a legal concept may not be understood by ordinary readers,
     briefly explain or unpack it using simple language
   - Do NOT add new legal meaning

7. DO NOT CHANGE OR WEAKEN:
   - Obligations: "አለበት", "ይገባል"
   - Prohibitions: "አይፈቀድም"
   - Conditions: "ከ...በስተቀር", "ካል..."
   - Legal roles: "ከሳሽ", "ተከሳሽ", "ባለቤት"
   - Numbers, dates, amounts
   - Legal references (articles, laws)

AFTER you generate the simplified sentence(s):

ANALYSIS TASK (MANDATORY):

Compare the original legal sentence and the simplified sentence.

Determine the SINGLE PRIMARY simplification strategy —
the one that contributes MOST to making the sentence easier to understand.

Even if multiple changes were made, you MUST choose the dominant one.

Choose EXACTLY ONE label from the list below:
- sentence_splitting
- vocabulary_simplification
- structure_reordering
- deletion
- definition_or_expansion

IMPORTANT RULES:
- DO NOT use "combined"
- DO NOT invent new labels
- DO NOT explain your choice
- Always choose ONE dominant strategy

OUTPUT FORMAT (STRICT — MUST FOLLOW EXACTLY):

SIMPLIFIED_SENTENCE:
<put the simplified sentence(s) here>

SIMPLIFICATION_TYPE:
<put exactly one label from the list here>


"""

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def regenerate_pair(legal_sentence, api_key, model_name):
    """Regenerate a single pair using specified API key and model
    
    Returns:
        tuple: (simplified_sentence, simplification_type, error)
        - simplified_sentence: str or None
        - simplification_type: str or None (one of: sentence_splitting, deletion, vocabulary_simplification, register_shift, structure_reordering, combined)
        - error: str or None
    """
    # Configure with specific API key
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = PROMPT_TEMPLATE.format(legal_sentence=legal_sentence)
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse the response to extract SIMPLIFIED_SENTENCE and SIMPLIFICATION_TYPE
        simplified = None
        simplification_type = None
        
        # Look for SIMPLIFIED_SENTENCE: marker
        if "SIMPLIFIED_SENTENCE:" in response_text:
            parts = response_text.split("SIMPLIFIED_SENTENCE:", 1)
            if len(parts) > 1:
                # Extract the simplified sentence (everything until SIMPLIFICATION_TYPE:)
                sentence_part = parts[1]
                if "SIMPLIFICATION_TYPE:" in sentence_part:
                    simplified = sentence_part.split("SIMPLIFICATION_TYPE:")[0].strip()
                else:
                    simplified = sentence_part.strip()
        
        # Look for SIMPLIFICATION_TYPE: marker
        if "SIMPLIFICATION_TYPE:" in response_text:
            parts = response_text.split("SIMPLIFICATION_TYPE:", 1)
            if len(parts) > 1:
                type_part = parts[1].strip()
                # Extract the first line (the type label)
                simplification_type = type_part.split('\n')[0].strip()
        
        # Fallback: if format wasn't followed, try to extract anyway
        if not simplified:
            # Try old format markers as fallback
            for marker in ["SIMPLIFIED SENTENCE:", "Simplified:", "**Simplified Sentence:**"]:
                if marker in response_text:
                    simplified = response_text.split(marker, 1)[1].strip()
                    # Remove any trailing SIMPLIFICATION_TYPE section
                    if "SIMPLIFICATION_TYPE:" in simplified:
                        simplified = simplified.split("SIMPLIFICATION_TYPE:")[0].strip()
                    break
        
        # If still no simplified text, use the whole response (minus markers)
        if not simplified:
            simplified = response_text
            # Remove known markers
            for marker in ["SIMPLIFIED_SENTENCE:", "SIMPLIFICATION_TYPE:", "SIMPLIFIED SENTENCE:", "Simplified:", "**Simplified Sentence:**"]:
                simplified = simplified.replace(marker, "").strip()
        
        # Validate simplification_type
        valid_types = ["sentence_splitting", "deletion", "vocabulary_simplification", "register_shift", "structure_reordering", "combined"]
        if simplification_type and simplification_type not in valid_types:
            # Try to normalize (remove extra whitespace, lowercase)
            simplification_type = simplification_type.strip().lower()
            if simplification_type not in valid_types:
                # Default to "combined" if invalid
                simplification_type = "combined"
        
        # If no type was extracted, default to "combined"
        if not simplification_type:
            simplification_type = "combined"
        
        return (simplified, simplification_type), None
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Categorize errors
        if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return (None, None), "RATE_LIMIT"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return (None, None), "NETWORK_TIMEOUT"
        elif "connection" in error_msg.lower() or "network" in error_msg.lower() or "socket" in error_msg.lower():
            return (None, None), "NETWORK_ERROR"
        elif "503" in error_msg or "502" in error_msg or "500" in error_msg:
            return (None, None), "SERVER_ERROR"
        else:
            return (None, None), f"{error_type}: {error_msg}"

def worker_thread(api_key, key_index, work_queue, results_list, results_lock, stats_dict, failed_pairs_list, failed_pairs_lock, model_index_dict, model_index_lock):
    """Worker thread that processes items from the queue with model fallback"""
    genai.configure(api_key=api_key)
    
    processed_count = 0
    success_count = 0
    error_count = 0
    
    # Network errors that should be retried
    NETWORK_ERRORS = ["NETWORK_TIMEOUT", "NETWORK_ERROR", "SERVER_ERROR"]
    MAX_NETWORK_RETRIES = 2  # Maximum retries for network errors per model
    
    while not shutdown_event.is_set():
        try:
            # Get work item from queue (timeout to check if queue is empty or shutdown)
            item = work_queue.get(timeout=1)
            
            # Poison pill to stop worker
            if item is None:
                break
            
            pair_index, pair_data = item
            legal = pair_data["legal_sentence"]
            
            # Process the pair with model fallback
            simplified = None
            simplification_type = None
            error = None
            model_used = None
            
            # Try each model in priority order until one works
            for model_idx, model_name in enumerate(MODEL_PRIORITY):
                # Check if this model has hit rate limit (skip if so)
                with model_index_lock:
                    if model_index_dict.get(model_name, 0) >= 3:  # Skip if rate limited 3+ times
                        continue
                
                model_used = model_name
                network_retry_count = 0
                
                while network_retry_count <= MAX_NETWORK_RETRIES:
                    result_tuple, error = regenerate_pair(legal, api_key, model_name)
                    
                    if error is None:
                        # Success! Extract simplified sentence and type
                        simplified, simplification_type = result_tuple
                        break
                    
                    # Handle rate limit - mark model and move to next
                    if error == "RATE_LIMIT":
                        with model_index_lock:
                            model_index_dict[model_name] = model_index_dict.get(model_name, 0) + 1
                        print(f"\n[Key {key_index+1}] ⚠️  Rate limit on {model_name}, switching to next model... (Pair {pair_index})")
                        error = None  # Reset error to try next model
                        break  # Break out of retry loop, try next model
                    
                    # Check if it's a retryable network error
                    if error in NETWORK_ERRORS:
                        network_retry_count += 1
                        if network_retry_count <= MAX_NETWORK_RETRIES:
                            wait_time = 10 * network_retry_count
                            print(f"\n[Key {key_index+1}] ⚠️  Network error on {model_name} (attempt {network_retry_count}/{MAX_NETWORK_RETRIES}): {error}")
                            print(f"   Retrying in {wait_time}s... (Pair {pair_index})")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Max network retries reached, try next model
                            print(f"\n[Key {key_index+1}] ⚠️  Network error persists on {model_name}, trying next model... (Pair {pair_index})")
                            error = None  # Reset to try next model
                            break
                    else:
                        # Non-retryable error (quota, API key issue, etc.) - try next model
                        print(f"\n[Key {key_index+1}] ⚠️  Error on {model_name}: {error}, trying next model... (Pair {pair_index})")
                        error = None  # Reset to try next model
                        break
                
                # If we got a successful result, break out of model loop
                if error is None and simplified is not None:
                    break
            
            # If we exhausted all models, mark as failed
            if simplified is None:
                error = "ALL_MODELS_EXHAUSTED"
                model_used = None
            
            # Create result
            result = {
                "legal_sentence": legal,
                "regenerated": error is None,
                "key_used": key_index,
                "pair_index": pair_index,
                "model_used": model_used
            }
            
            if error:
                result["simplified_sentence"] = ""  # Empty on failure
                result["simplification_type"] = None  # No type on failure
                result["error"] = error
                error_count += 1
                
                # Save to failed pairs list for later retry
                with failed_pairs_lock:
                    failed_pairs_list.append({
                        "legal_sentence": legal,
                        "pair_index": pair_index,
                        "error": error,
                        "key_used": key_index,
                        "model_used": model_used
                    })
            else:
                result["simplified_sentence"] = simplified
                result["simplification_type"] = simplification_type
                success_count += 1
            
            # Add to results (thread-safe)
            with results_lock:
                results_list.append(result)
            
            # Save incrementally after each result (outside lock to avoid blocking)
            save_results_incremental(results_list, results_lock, stats_dict)
            
            processed_count += 1
            
            # Print progress with error count
            with stats_dict['lock']:
                stats_dict['total_processed'] += 1
                stats_dict['total_success'] += (1 if error is None else 0)
                total = stats_dict['total_processed']
                success = stats_dict['total_success']
                failed = total - success
                
                if error:
                    error_short = error[:40] if len(error) > 40 else error
                    print(f"[Key {key_index+1}] [{total}/{stats_dict['total_items']}] ✓ {success} | ✗ {failed} | Last: {error_short}", end='\r', flush=True)
                else:
                    print(f"[Key {key_index+1}] [{total}/{stats_dict['total_items']}] ✓ {success} | ✗ {failed} | Processing...", end='\r', flush=True)
            
            # Wait 5 seconds after response (to stay under rate limit)
            time.sleep(POST_REQUEST_WAIT)
            
            # Mark task as done
            work_queue.task_done()
            
        except queue.Empty:
            # Queue is empty, check if we should continue
            continue
        except Exception as e:
            print(f"\n[Key {key_index+1}] ❌ Worker exception: {type(e).__name__}: {e}")
            error_count += 1
            work_queue.task_done()
    
    return {
        "key_index": key_index,
        "processed": processed_count,
        "success": success_count,
        "errors": error_count
    }

# ============================================================================
# INCREMENTAL SAVING FUNCTIONS
# ============================================================================

def save_results_incremental(results_list, results_lock, stats_dict):
    """Save results to JSON file incrementally (thread-safe)"""
    batch_dir = Path("Dataset")
    output_file = batch_dir / "all_regenerated_pairs.json"
    failed_file = Path("Dataset/final_dataset") / "failed_pairs_for_retry.json"
    
    # Create a copy of results for saving (to minimize lock time)
    with results_lock:
        results_copy = sorted(results_list.copy(), key=lambda x: x.get('pair_index', 0))
        failed_copy = [r for r in results_copy if not r.get('regenerated', False)]
    
    # Save all results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save results: {e}")
    
    # Save failed pairs if any
    if failed_copy:
        try:
            failed_sorted = sorted(failed_copy, key=lambda x: x.get('pair_index', 0))
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_sorted, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save failed pairs: {e}")

# Global variables for signal handling
shutdown_event = threading.Event()
results_list_global = None
results_lock_global = None
stats_dict_global = None
failed_pairs_list_global = None
failed_pairs_lock_global = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Interrupt received! Saving current results...")
    shutdown_event.set()
    
    # Save current results
    if results_list_global and results_lock_global:
        save_results_incremental(results_list_global, results_lock_global, stats_dict_global)
        
        # Also save failed pairs
        if failed_pairs_list_global and failed_pairs_lock_global:
            failed_file = Path("Dataset/final_dataset") / "failed_pairs_for_retry.json"
            with failed_pairs_lock_global:
                failed_copy = sorted(failed_pairs_list_global.copy(), key=lambda x: x.get('pair_index', 0))
            try:
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_copy, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️  Warning: Could not save failed pairs: {e}")
        
        with results_lock_global:
            total = len(results_list_global)
            success = sum(1 for r in results_list_global if r.get('regenerated'))
        
        print(f"\n✅ Saved {total} results ({success} successful) to all_regenerated_pairs.json")
        print("   You can resume later or retry failed pairs.\n")
    
    sys.exit(0)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("PARALLEL GEMINI REGENERATION (MODEL FALLBACK SYSTEM)")
    print("=" * 80)
    print(f"\nUsing {len(API_KEYS)} API keys")
    print(f"Models (with fallback): {len(MODEL_PRIORITY)} models")
    for i, model in enumerate(MODEL_PRIORITY, 1):
        print(f"  {i}. {model}")
    print(f"Wait time after each response: {POST_REQUEST_WAIT} seconds")
    
    # Check for placeholder keys
    placeholder_keys = [i for i, key in enumerate(API_KEYS) if "YOUR_API_KEY" in key]
    if placeholder_keys:
        print(f"\n⚠️  WARNING: {len(placeholder_keys)} API keys are placeholders!")
        print(f"   Please replace keys at indices: {placeholder_keys}")
        return
    
    # Step 1: Load already processed pairs to skip them
    processed_file = Path("Dataset/all_regenerated_pairs1.json")
    processed_legal_sentences = set()
    if processed_file.exists():
        print(f"\n📋 Loading already processed pairs from: {processed_file.name}")
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        for item in processed_data:
            if item.get('regenerated', False):
                processed_legal_sentences.add(item.get('legal_sentence', ''))
        print(f"   Found {len(processed_legal_sentences)} already processed pairs (will skip)")
    else:
        print(f"\n📋 No existing processed file found, starting fresh")
    
    # Step 2: Load top_800_priority.json first
    priority_file = Path("Dataset/final_dataset/top_800_priority.json")
    priority_pairs = []
    priority_legal_sentences = set()
    
    if priority_file.exists():
        print(f"\n📦 Phase 1: Loading priority pairs from: {priority_file.name}")
        with open(priority_file, 'r', encoding='utf-8') as f:
            priority_data = json.load(f)
        
        for item in priority_data:
            legal = item.get('legal', '')
            if legal and legal not in processed_legal_sentences:
                priority_legal_sentences.add(legal)
                priority_pairs.append({
                    "legal_sentence": legal,
                    "simplified_sentence": item.get('simplified', '')  # Original simplified for reference
                })
        
        print(f"   Loaded {len(priority_pairs)} priority pairs (after skipping already processed)")
    else:
        print(f"\n⚠️  Priority file not found: {priority_file}")
    
    # Step 3: Load final.json and exclude priority pairs
    final_file = Path("Dataset/final_dataset/final.json")
    remaining_pairs = []
    
    if final_file.exists():
        print(f"\n📦 Phase 2: Loading remaining pairs from: {final_file.name}")
        with open(final_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        for item in final_data:
            legal = item.get('legal_sentence', '')
            # Skip if already in priority or already processed
            if legal and legal not in priority_legal_sentences and legal not in processed_legal_sentences:
                remaining_pairs.append(item)
        
        print(f"   Loaded {len(remaining_pairs)} remaining pairs (after excluding priority and processed)")
    else:
        print(f"\n⚠️  Final file not found: {final_file}")
    
    # Combine all pairs to process (priority first, then remaining)
    all_pairs_to_process = priority_pairs + remaining_pairs
    total_items = len(all_pairs_to_process)
    
    if total_items == 0:
        print("\n✅ No pairs to process! All pairs are already done.")
        return
    
    print(f"\n📊 Total pairs to process: {total_items}")
    print(f"   - Priority pairs: {len(priority_pairs)}")
    print(f"   - Remaining pairs: {len(remaining_pairs)}")
    
    # Create work queue
    work_queue = queue.Queue()
    for pair_id, pair in enumerate(all_pairs_to_process):
        work_queue.put((pair_id, pair))
    
    # Shared results list and stats
    results_list = []
    results_lock = threading.Lock()
    failed_pairs_list = []  # List to store failed pairs for retry
    failed_pairs_lock = threading.Lock()
    model_index_dict = {}  # Track rate-limited models (model_name -> count)
    model_index_lock = threading.Lock()
    stats_dict = {
        'total_items': total_items,
        'total_processed': 0,
        'total_success': 0,
        'lock': threading.Lock()
    }
    
    # Set up global variables for signal handling
    global results_list_global, results_lock_global, stats_dict_global
    global failed_pairs_list_global, failed_pairs_lock_global
    results_list_global = results_list
    results_lock_global = results_lock
    stats_dict_global = stats_dict
    failed_pairs_list_global = failed_pairs_list
    failed_pairs_lock_global = failed_pairs_lock
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize output file (create empty file to show it's started)
    output_file = Path("Dataset/all_regenerated_pairs.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print(f"📝 Results will be saved incrementally to: {output_file}")
    
    # Start worker threads (one per API key)
    print(f"\n🚀 Starting {len(API_KEYS)} worker threads...\n")
    print("⏳ Waiting 10 seconds before starting to avoid initial rate limits...")
    time.sleep(10)
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        # Submit all workers with staggered start
        futures = []
        for key_idx, api_key in enumerate(API_KEYS):
            # Stagger start: wait 2 seconds between each worker
            if key_idx > 0:
                time.sleep(2)
                print(f"   Starting worker {key_idx+1}/{len(API_KEYS)}...")
            
            future = executor.submit(
                worker_thread,
                api_key,
                key_idx,
                work_queue,
                results_list,
                results_lock,
                stats_dict,
                failed_pairs_list,
                failed_pairs_lock,
                model_index_dict,
                model_index_lock
            )
            futures.append(future)
        
        # Wait for all work to complete (or until shutdown)
        try:
            work_queue.join()
        except KeyboardInterrupt:
            # Already handled by signal handler, but ensure we break
            pass
        
        # Send poison pills to stop workers
        for _ in range(len(API_KEYS)):
            work_queue.put(None)
        
        # Wait for all workers to finish
        worker_stats = []
        for future in as_completed(futures):
            try:
                stats = future.result()
                worker_stats.append(stats)
            except Exception as e:
                print(f"\n❌ Worker error: {e}")
    
    total_elapsed = time.time() - start_time
    
    # Final save (results are already being saved incrementally, but do a final save)
    print(f"\n{'='*80}")
    print("FINAL SAVE")
    print(f"{'='*80}\n")
    
    # Sort results by pair_index to maintain order
    with results_lock:
        results_list.sort(key=lambda x: x.get('pair_index', 0))
        save_results_incremental(results_list, results_lock, stats_dict)
    
    output_file = Path("Dataset/all_regenerated_pairs.json")
    
    with results_lock:
        success = sum(1 for r in results_list if r.get('regenerated'))
        total = len(results_list)
    
    print(f"✅ {output_file}: {success}/{total} successful")
    
    # Save failed pairs for later retry
    if failed_pairs_list:
        failed_file = Path("Dataset/final_dataset") / "failed_pairs_for_retry.json"
        # Sort by pair_index
        failed_pairs_list.sort(key=lambda x: x.get('pair_index', 0))
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_pairs_list, f, ensure_ascii=False, indent=2)
        
        # Group errors by type
        error_types = {}
        for pair in failed_pairs_list:
            error = pair.get('error', 'UNKNOWN')
            error_type = error.split(':')[0] if ':' in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print(f"\n❌ {failed_file.name}: {len(failed_pairs_list)} failed pairs saved for retry")
        print(f"   Error breakdown:")
        for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {err_type}: {count}")
    
    # Summary
    total_success = sum(1 for r in results_list if r.get('regenerated'))
    total_errors = len(results_list) - total_success
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total pairs processed: {len(results_list)}")
    print(f"Successfully regenerated: {total_success} ({100*total_success/len(results_list):.1f}%)")
    print(f"Errors: {total_errors}")
    if failed_pairs_list:
        print(f"Failed pairs saved for retry: {len(failed_pairs_list)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Average time per pair: {total_elapsed/len(results_list):.1f} seconds")
    
    # Worker statistics
    print(f"\nWorker Statistics:")
    for stats in worker_stats:
        print(f"  Key {stats['key_index']+1}: {stats['processed']} processed, {stats['success']} success, {stats['errors']} errors")

if __name__ == "__main__":
    main()
