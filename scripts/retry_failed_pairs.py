# retry_failed_pairs.py
# Script to retry failed pairs from failed_pairs_for_retry.json
import json
import time
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import queue
import threading

# ============================================================================
# CONFIGURATION (Same as regenerate_parallel.py)
# ============================================================================

API_KEYS = [
    "REPLACE WITH YOUR API KEY HERE",  # Key 1
]

MODEL_NAME = "gemini-2.5-flash"
POST_REQUEST_WAIT = 8.0

# Same prompt template as regenerate_parallel.py
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
   - Do NOT delete conditions or obligations

4. VOCABULARY SIMPLIFICATION:
   - Replace archaic legal wording with common Amharic:
     * "ይገደዳል" → "አለበት"
     * "ተፈጻሚነት" → "መፈጸም" or "መስራት"

5. STRUCTURE SIMPLIFICATION:
   - Break nested clauses into linear order
   - Make passive constructions active when possible
   - Make implicit actors explicit when clear

6. DO NOT CHANGE OR WEAKEN:
   - Obligations: "አለበት", "ይገባል"
   - Prohibitions: "አይፈቀድም"
   - Conditions: "ከ...በስተቀር", "ካል..."
   - Legal roles: "ከሳሽ", "ተከሳሽ", "ባለቤት"
   - Numbers, dates, amounts
   - Legal references (articles, laws)

AFTER you generate the simplified sentence(s):

ANALYSIS TASK (MANDATORY):
Compare the original legal sentence and the simplified sentence.
Determine the PRIMARY simplification strategy used.

Choose ONE label from the list below:
- sentence_splitting
- deletion
- vocabulary_simplification
- register_shift
- structure_reordering
- combined

RULES FOR LABELING:
- If two or more strategies are clearly used, choose "combined"
- Choose the dominant strategy, not minor edits

OUTPUT FORMAT (STRICT — MUST FOLLOW EXACTLY):

SIMPLIFIED_SENTENCE:
<put the simplified sentence(s) here>

SIMPLIFICATION_TYPE:
<put exactly one label from the list here>

"""

def regenerate_pair(legal_sentence, api_key, model_name):
    """Regenerate a single pair using specified API key and model
    
    Returns:
        tuple: (simplified_sentence, simplification_type, error)
        - simplified_sentence: str or None
        - simplification_type: str or None (one of: sentence_splitting, deletion, vocabulary_simplification, register_shift, structure_reordering, combined)
        - error: str or None
    """
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

def worker_thread(api_key, key_index, work_queue, results_list, results_lock, stats_dict, failed_pairs_list, failed_pairs_lock):
    """Worker thread that processes items from the queue"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    processed_count = 0
    success_count = 0
    error_count = 0
    
    NETWORK_ERRORS = ["NETWORK_TIMEOUT", "NETWORK_ERROR", "SERVER_ERROR"]
    MAX_RETRIES = 3
    
    while True:
        try:
            item = work_queue.get(timeout=1)
            
            if item is None:
                break
            
            pair_index, pair_data = item
            legal = pair_data["legal_sentence"]
            
            simplified = None
            simplification_type = None
            error = None
            retry_count = 0
            
            while retry_count <= MAX_RETRIES:
                result_tuple, error = regenerate_pair(legal, api_key, MODEL_NAME)
                
                if error is None:
                    # Success! Extract simplified sentence and type
                    simplified, simplification_type = result_tuple
                    break
                
                if error in NETWORK_ERRORS:
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        wait_time = 10 * retry_count
                        print(f"\n[Key {key_index+1}] ⚠️  Network error (attempt {retry_count}/{MAX_RETRIES}): {error}")
                        print(f"   Retrying in {wait_time}s... (Pair {pair_index})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n[Key {key_index+1}] ❌ Failed after {MAX_RETRIES} retries: {error} (Pair {pair_index})")
                        break
                elif error == "RATE_LIMIT":
                    if retry_count == 0:
                        print(f"\n[Key {key_index+1}] ⚠️  Rate limit hit, waiting 60s... (Pair {pair_index})")
                        time.sleep(60)
                        retry_count += 1
                        continue
                    else:
                        print(f"\n[Key {key_index+1}] ❌ Rate limit persists after retry (Pair {pair_index})")
                        break
                else:
                    print(f"\n[Key {key_index+1}] ❌ Error: {error} (Pair {pair_index})")
                    break
            
            result = {
                "legal_sentence": legal,
                "regenerated": error is None,
                "key_used": key_index,
                "pair_index": pair_index
            }
            
            if error:
                result["simplified_sentence"] = ""
                result["simplification_type"] = None
                result["error"] = error
                result["retry_count"] = retry_count
                error_count += 1
                
                with failed_pairs_lock:
                    failed_pairs_list.append({
                        "legal_sentence": legal,
                        "pair_index": pair_index,
                        "error": error,
                        "key_used": key_index,
                        "retry_count": retry_count
                    })
            else:
                result["simplified_sentence"] = simplified
                result["simplification_type"] = simplification_type
                success_count += 1
            
            with results_lock:
                results_list.append(result)
            
            processed_count += 1
            
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
            
            time.sleep(POST_REQUEST_WAIT)
            work_queue.task_done()
            
        except queue.Empty:
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

def main():
    failed_file = Path("Dataset/final_dataset/failed_pairs_for_retry.json")
    
    if not failed_file.exists():
        print("❌ Failed pairs file not found!")
        print(f"   Looking for: {failed_file}")
        return
    
    print("=" * 80)
    print("RETRY FAILED PAIRS")
    print("=" * 80)
    
    with open(failed_file, 'r', encoding='utf-8') as f:
        failed_pairs = json.load(f)
    
    print(f"\n📦 Loaded {len(failed_pairs)} failed pairs to retry")
    print(f"Using {len(API_KEYS)} API keys")
    print(f"Model: {MODEL_NAME}")
    
    work_queue = queue.Queue()
    
    for pair in failed_pairs:
        work_queue.put((pair.get('pair_index', 0), pair))
    
    total_items = len(failed_pairs)
    
    results_list = []
    results_lock = threading.Lock()
    failed_pairs_list = []
    failed_pairs_lock = threading.Lock()
    stats_dict = {
        'total_items': total_items,
        'total_processed': 0,
        'total_success': 0,
        'lock': threading.Lock()
    }
    
    print(f"\n🚀 Starting {len(API_KEYS)} worker threads...\n")
    print("⏳ Waiting 10 seconds before starting to avoid initial rate limits...")
    time.sleep(10)
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
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
                failed_pairs_lock
            )
            futures.append(future)
        
        work_queue.join()
        
        for _ in range(len(API_KEYS)):
            work_queue.put(None)
        
        worker_stats = []
        for future in as_completed(futures):
            try:
                stats = future.result()
                worker_stats.append(stats)
            except Exception as e:
                print(f"\n❌ Worker error: {e}")
    
    total_elapsed = time.time() - start_time
    
    results_list.sort(key=lambda x: x.get('pair_index', 0))
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    batch_dir = Path("Dataset/final_dataset")
    
    # Save retry results
    retry_output = batch_dir / "retry_results.json"
    with open(retry_output, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    
    success = sum(1 for r in results_list if r.get('regenerated'))
    print(f"✅ {retry_output.name}: {success}/{len(results_list)} successful")
    
    if failed_pairs_list:
        still_failed = batch_dir / "still_failed_pairs.json"
        failed_pairs_list.sort(key=lambda x: x.get('pair_index', 0))
        with open(still_failed, 'w', encoding='utf-8') as f:
            json.dump(failed_pairs_list, f, ensure_ascii=False, indent=2)
        print(f"❌ {still_failed.name}: {len(failed_pairs_list)} still failed")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total pairs retried: {len(results_list)}")
    print(f"Successfully regenerated: {success} ({100*success/len(results_list):.1f}%)")
    print(f"Still failed: {len(failed_pairs_list)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()

