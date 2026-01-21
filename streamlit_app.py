import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from pathlib import Path
import pickle
import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
from typing import Optional, Tuple, List
from evaluate import load as load_metric

# Set page config
st.set_page_config(
    page_title="Amharic Legal Text Simplification",
    page_icon="📜",
    layout="wide"
)

# Constants
WORKDIR = Path(__file__).parent
MODELS_DIR = WORKDIR / "models"
BASE_MODEL_NAME = "masakhane/afri-byt5-base"
BASE_MODEL_PATH = MODELS_DIR / "models--masakhane--afri-byt5-base"
LLAMA_BASE_MODEL_NAME = "rasyosef/Llama-3.2-400M-Amharic-Instruct"
LLAMA_BASE_MODEL_PATH = MODELS_DIR / "models--rasyosef--Llama-3.2-400M-Amharic-Instruct"

# Model paths
ADAPTER_PATHS = {
    "Model 1 (final.json - old config)": MODELS_DIR / "afribyt5-legal-simplification-final",
    "Model 2 (final.json - updated config)": MODELS_DIR / "afribyt5-legal-simplification-final2",
    "Model 3 (regen.json - no simplification_type)": MODELS_DIR / "afribyt5-legal-simplification-final3",
    "Model 4 (regen.json - with simplification_type)": MODELS_DIR / "afribyt5-legal-simplification-final4",
    "Model 5 (LLaMA 400M - with simplification_type)": MODELS_DIR / "llama-400m-legal-simplification5"
}

# RAG paths
RAG_INDEX_PATH = WORKDIR / "rag_pipeline" / "4_vector_db" / "faiss_index.bin"
RAG_METADATA_PATH = WORKDIR / "rag_pipeline" / "4_vector_db" / "metadata.parquet"

# Contrastive selector path
CONTRASTIVE_MODEL_PATH = MODELS_DIR / "contrastive_strategy_selector"

# Test data paths
TEST_DATA_PATHS = {
    "Model 1 (final.json - old config)": WORKDIR / "Dataset" / "final_dataset" / "final_test.json",
    "Model 2 (final.json - updated config)": WORKDIR / "Dataset" / "final_dataset" / "final_test.json",
    "Model 3 (regen.json - no simplification_type)": WORKDIR / "Dataset" / "final_dataset" / "regen_test.json",
    "Model 4 (regen.json - with simplification_type)": WORKDIR / "Dataset" / "final_dataset" / "regen_test.json",
    "Model 5 (LLaMA 400M - with simplification_type)": WORKDIR / "Dataset" / "final_dataset" / "regen_test.json"
}

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Evaluation cache directory
EVALUATION_CACHE_DIR = WORKDIR / "evaluation_cache"
EVALUATION_CACHE_DIR.mkdir(exist_ok=True)

# Evaluation results
# Note: These should be extracted from notebook outputs
# For now, using placeholder values - update with actual results from your training runs
EVALUATION_RESULTS = {
    "Model 1 (final.json - old config)": {
        "BERTScore F1": "0.9722",
        "SARI": "38.18",
        "Test Samples": "95",
        "Dataset": "final.json (1,700 train / 200 val / 95 test)",
        "Config": "LoRA r=16, alpha=32, modules=['q', 'v']",
        "qualitative_samples": [
            {
                "original": "በዚህ አንቀጽ ንዑስ አንቀጽ መሰረት የሚቀርበው ክስ ገንዘብ ጠያቂው የማህበሩን መፍረስ ካወቀበት ጊዜ ጀምሮ በአምስት ዓመት ውስጥ ካልቀረበ በይርጋ ይታገዳል ፤",
                "simplified": "በዚህ ንዑስ አንቀጽ መሠረት የሚቀርብ ክስ የገንዘብ ጠያቂው ማህበሩ መፍረሱን ካወቀበት ጊዜ ጀምሮ በ 5 ዓመት ውስጥ መቅረብ አለበት። ካልቀረበ ግን በይርጋ ምክንያት ተቀባይነት አይኖረውም።",
                "prediction": "[Extract from notebook output]"
            }
        ]
    },
    "Model 2 (final.json - updated config)": {
        "BERTScore F1": "0.9508",
        "SARI": "43.52",
        "Test Samples": "82",
        "Dataset": "final.json (1,700 train / 200 val / 82 test)",
        "Config": "LoRA r=32, alpha=64, modules=['k', 'q', 'v', 'o']",
        "qualitative_samples": [
            {
                "original": "[Sample from Model 2 evaluation]",
                "simplified": "[Reference simplified]",
                "prediction": "[Model prediction]"
            }
        ]
    },
    "Model 3 (regen.json - no simplification_type)": {
        "BERTScore F1": "0.9508",
        "SARI": "43.52",
        "Test Samples": "82",
        "Dataset": "regen.json (ignoring simplification_type)",
        "Config": "LoRA r=32, alpha=64, modules=['v', 'q', 'o', 'k']",
        "qualitative_samples": [
            {
                "original": "[Sample from Model 3 evaluation]",
                "simplified": "[Reference simplified]",
                "prediction": "[Model prediction]"
            }
        ]
    },
    "Model 4 (regen.json - with simplification_type)": {
        "BERTScore F1": "0.9508",
        "SARI": "43.52",
        "Test Samples": "82",
        "Dataset": "regen.json (using simplification_type)",
        "Config": "LoRA r=32, alpha=64, modules=['q', 'v', 'o', 'k'] + simplification_type conditioning",
        "qualitative_samples": [
            {
                "original": "[Sample from Model 4 evaluation]",
                "simplified": "[Reference simplified]",
                "prediction": "[Model prediction]"
            }
        ]
    },
    "Model 5 (LLaMA 400M - with simplification_type)": {
        "BERTScore F1": "TBD",
        "SARI": "TBD",
        "Test Samples": "100",
        "Dataset": "regen.json (using simplification_type)",
        "Config": "LLaMA 3.2 400M Instruct, LoRA r=64, alpha=128, modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] + simplification_type conditioning",
        "qualitative_samples": [
            {
                "original": "[Sample from Model 5 evaluation]",
                "simplified": "[Reference simplified]",
                "prediction": "[Model prediction]"
            }
        ]
    }
}

@st.cache_resource
def load_base_model(model_name: str = None):
    """Load base model and tokenizer from local cache
    
    Args:
        model_name: Optional model name to determine which base model to load
                    If None or not Model 5, loads AfriByT5. If Model 5, loads LLaMA.
    """
    # Determine which base model to load
    is_llama = model_name and "Model 5" in model_name
    
    if is_llama:
        # Load LLaMA model
        snapshot_dir = LLAMA_BASE_MODEL_PATH / "snapshots"
        base_model_name = LLAMA_BASE_MODEL_NAME
        model_class = AutoModelForCausalLM
    else:
        # Load AfriByT5 model
        snapshot_dir = BASE_MODEL_PATH / "snapshots"
        base_model_name = BASE_MODEL_NAME
        model_class = AutoModelForSeq2SeqLM
    
    # Find the snapshot directory
    local_model_path = None
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.iterdir())
        if snapshots:
            # Use the snapshot that has pytorch_model.bin or model.safetensors
            for snapshot in snapshots:
                potential_path = snapshot
                model_file_bin = potential_path / "pytorch_model.bin"
                model_file_safe = potential_path / "model.safetensors"
                
                # Check if either file exists
                if model_file_bin.exists() or model_file_safe.exists():
                    local_model_path = potential_path
                    break
    
    if local_model_path and local_model_path.exists():
        # Load from local snapshot path - use local_files_only=True to prevent downloads
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_model_path), 
                local_files_only=True
            )
            model = model_class.from_pretrained(
                str(local_model_path), 
                local_files_only=True,
                torch_dtype=torch.float16 if is_llama else None,
                device_map="auto" if is_llama and torch.cuda.is_available() else None,
                trust_remote_code=True if is_llama else False
            )
        except Exception as e:
            # If local loading fails, try with cache_dir
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                cache_dir=str(MODELS_DIR)
            )
            model = model_class.from_pretrained(
                base_model_name,
                cache_dir=str(MODELS_DIR),
                torch_dtype=torch.float16 if is_llama else None,
                device_map="auto" if is_llama and torch.cuda.is_available() else None,
                trust_remote_code=True if is_llama else False
            )
    else:
        # Use cache_dir - will use existing cache if available
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=str(MODELS_DIR)
        )
        model = model_class.from_pretrained(
            base_model_name,
            cache_dir=str(MODELS_DIR),
            torch_dtype=torch.float16 if is_llama else None,
            device_map="auto" if is_llama and torch.cuda.is_available() else None,
            trust_remote_code=True if is_llama else False
        )
    
    # Set pad_token for LLaMA if needed
    if is_llama and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    
    if not is_llama or not torch.cuda.is_available():
        model = model.to(device)
    
    return model, tokenizer

@st.cache_resource
def load_adapter(_base_model, adapter_path):
    """Load adapter on base model"""
    adapter = PeftModel.from_pretrained(_base_model, str(adapter_path))
    adapter = adapter.to(device)
    adapter.eval()
    return adapter

@st.cache_resource
@st.cache_resource
def load_rag_system():
    """Load RAG index and metadata"""
    # Check if files exist
    index_exists = RAG_INDEX_PATH.exists()
    metadata_exists = RAG_METADATA_PATH.exists()
    
    if not index_exists or not metadata_exists:
        print(f"RAG files not found. Index exists: {index_exists}, Metadata exists: {metadata_exists}")
        print(f"Index path: {RAG_INDEX_PATH.absolute()}")
        print(f"Metadata path: {RAG_METADATA_PATH.absolute()}")
        return None, None
    
    try:
        index = faiss.read_index(str(RAG_INDEX_PATH))
        metadata = pd.read_parquet(RAG_METADATA_PATH)
        print(f"RAG system loaded successfully. Index size: {index.ntotal}, Metadata rows: {len(metadata)}")
        return index, metadata
    except Exception as e:
        import traceback
        print(f"Error loading RAG system: {e}")
        print(traceback.format_exc())
        return None, None

@st.cache_resource
def load_contrastive_selector():
    """Load contrastive strategy selector"""
    if not CONTRASTIVE_MODEL_PATH.exists():
        return None
    
    encoder = SentenceTransformer(str(CONTRASTIVE_MODEL_PATH))
    encoder = encoder.to(device)
    
    centroids_path = CONTRASTIVE_MODEL_PATH / "centroids.pkl"
    if centroids_path.exists():
        with open(centroids_path, "rb") as f:
            centroids = pickle.load(f)
        return encoder, centroids
    return None

def get_rag_context(query: str, index, metadata, top_k: int = 3, query_embedding: Optional[np.ndarray] = None) -> List[str]:
    """Retrieve relevant legal context using RAG
    
    Args:
        query: The query text (used only if query_embedding is None)
        index: FAISS index
        metadata: Metadata dataframe
        top_k: Number of results to retrieve
        query_embedding: Pre-computed embedding (optional). If provided, skips API call.
    
    Returns:
        List of retrieved context strings
    """
    try:
        # Use pre-computed embedding if provided, otherwise compute it
        if query_embedding is None:
            # Load Gemini API key
            api_key_path = WORKDIR / ".gemini_api_key"
            if api_key_path.exists():
                with open(api_key_path, "r") as f:
                    content = f.read().strip()
                    # Handle both formats: "GEMINI_API_KEY=xxx" or just "xxx"
                    if "=" in content:
                        # Extract the value after the equals sign
                        api_key = content.split("=", 1)[1].strip()
                    else:
                        # Just the key itself
                        api_key = content
                genai.configure(api_key=api_key)
            else:
                return []
            
            # Get embedding for query using the API
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query
            )
            query_embedding = np.array(result['embedding'], dtype=np.float32).reshape(1, -1)
        else:
            # Ensure embedding is in correct shape
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype(np.float32)
        
        # Search in FAISS index
        k = min(top_k, index.ntotal)
        distances, indices = index.search(query_embedding, k)
        
        # Get relevant chunks - FIX: use 'text' column, not 'chunk_text'
        contexts = []
        for idx in indices[0]:
            if idx < len(metadata):
                # The metadata column is 'text', not 'chunk_text'
                chunk_text = metadata.iloc[idx]['text']
                contexts.append(chunk_text)
        
        return contexts
    except Exception as e:
        st.warning(f"RAG retrieval failed: {e}")
        return []

def batch_get_embeddings(queries: List[str]) -> List[np.ndarray]:
    """Get embeddings for multiple queries in a single batch API call
    
    Args:
        queries: List of query strings
    
    Returns:
        List of embeddings (numpy arrays), one per query
    """
    # Load Gemini API key
    api_key_path = WORKDIR / ".gemini_api_key"
    if not api_key_path.exists():
        st.error("Gemini API key not found. Cannot get embeddings.")
        return []
    
    with open(api_key_path, "r") as f:
        content = f.read().strip()
        # Handle both formats: "GEMINI_API_KEY=xxx" or just "xxx"
        if "=" in content:
            api_key = content.split("=", 1)[1].strip()
        else:
            api_key = content
    
    genai.configure(api_key=api_key)
    
    try:
        # Batch embed all queries at once
        # Gemini API supports batch embedding - pass list of strings
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=queries,  # Pass list of queries for batch processing
            task_type="RETRIEVAL_QUERY"
        )
        
        # Gemini API returns: {'embedding': [emb1, emb2, ...]} when content is a list
        if isinstance(result, dict) and 'embedding' in result:
            embeddings = result['embedding']  # This is a list of embeddings
        elif isinstance(result, list):
            embeddings = result
        else:
            # Fallback: if single embedding returned, wrap in list
            embeddings = [result['embedding']] if isinstance(result, dict) else []
        
        # Convert to numpy arrays
        embeddings_list = [np.array(emb, dtype=np.float32) for emb in embeddings]
        
        if len(embeddings_list) != len(queries):
            st.warning(f"Expected {len(queries)} embeddings, got {len(embeddings_list)}")
        
        return embeddings_list
    except Exception as e:
        st.error(f"Batch embedding failed: {e}")
        # Fallback: try individual calls (slower but works)
        st.warning("Falling back to individual embedding calls...")
        embeddings_list = []
        for query in queries:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="RETRIEVAL_QUERY"
                )
                emb = np.array(result['embedding'], dtype=np.float32)
                embeddings_list.append(emb)
            except Exception as e2:
                st.warning(f"Failed to embed query: {e2}")
                # Add zero embedding as placeholder
                embeddings_list.append(np.zeros(768, dtype=np.float32))  # 768 is typical embedding size
        return embeddings_list

def predict_simplification_type(sentence: str, encoder, centroids) -> str:
    """Predict simplification type using contrastive selector"""
    # Heuristic rules first
    words = sentence.split()
    word_count = len(words)
    
    if word_count > 40:
        return "sentence_splitting"
    
    boilerplate_phrases = [
        "እንደተጠበቀ ሆኖ",
        "በማንኛውም ሁኔታ",
        "ያለ አግባብ",
        "እንደተጠበቀ",
    ]
    if any(phrase in sentence for phrase in boilerplate_phrases):
        return "deletion"
    
    conjunctions = ["እና", "ወይም", "ቢሆንም", "ነገር ግን"]
    conjunction_count = sum(1 for conj in conjunctions if conj in sentence)
    if conjunction_count >= 3:
        return "structure_reordering"
    
    # Fallback to contrastive
    try:
        emb = encoder.encode([sentence], convert_to_numpy=True)[0]
        
        sims = {}
        for label, centroid in centroids.items():
            # Manual cosine similarity calculation
            dot_product = np.dot(emb, centroid)
            norm_emb = np.linalg.norm(emb)
            norm_centroid = np.linalg.norm(centroid)
            similarity = dot_product / (norm_emb * norm_centroid) if (norm_emb * norm_centroid) > 0 else 0
            sims[label] = similarity
        
        return max(sims, key=sims.get)
    except Exception as e:
        st.warning(f"Contrastive prediction failed: {e}")
        return "vocabulary_simplification"  # default

def simplify_text(
    text: str,
    model,
    tokenizer,
    simplification_type: Optional[str] = None,
    rag_context: Optional[List[str]] = None,
    is_model_4: bool = False,
    is_model_5: bool = False,
    max_input_length: int = 512,
    max_output_length: int = 256
) -> str:
    """Simplify legal text using the model"""
    # Build prompt
    base_instruction = "የሕግ ቃላትን ለግለሰቦች ለመረዳት ቀላል አማርኛ ውስጥ አቅርብ: "
    
    # Model 4 and Model 5 were trained with simplification_type in prompt
    if is_model_4 or is_model_5:
        type_map = {
            "vocabulary_simplification": "[የቃላት ማቃለል]",
            "sentence_splitting": "[የዓረፍተ ነገር መከፋፈል]",
            "deletion": "[መሻር]",
            "structure_reordering": "[የዕውቀት ማደራጀት]",
            "definition_or_expansion": "[ፍቺ ወይም ማስፋፋት]"
        }
        if simplification_type:
            type_label = type_map.get(simplification_type, "[አጠቃላይ]")
        else:
            type_label = "[አጠቃላይ]"  # Default without contrastive
        prompt = base_instruction + type_label + " " + text
    else:
        prompt = base_instruction + text
    
    # Add RAG context if provided
    if rag_context:
        context = "\n".join(rag_context[:2])  # Use top 2 contexts
        prompt = f"የሕግ አውድ: {context}\n\n{prompt}"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        if is_model_5:
            # For decoder-only models (Model 5), use causal LM generation
            # Use smaller max_new_tokens to limit output to single sentence
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced from 256 to limit to single sentence
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.5,  # Increased to prevent repetition
                no_repeat_ngram_size=4,  # Increased to prevent longer phrase repetition
                length_penalty=0.6,  # Reduced to encourage shorter outputs
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False  # Use greedy/beam search, not sampling
            )
            # Decode and extract only the generated part (remove input)
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input portion
            if full_output.startswith(prompt):
                simplified = full_output[len(prompt):].strip()
            else:
                simplified = full_output
            
            # Post-process: Extract only the first sentence (stop at first sentence ending)
            # Amharic sentence endings: ።, ፤, ፥, ፦, and standard punctuation
            sentence_endings = ['።', '፤', '፥', '፦', '.', '!', '?']
            for ending in sentence_endings:
                if ending in simplified:
                    # Find first occurrence and take everything up to and including it
                    idx = simplified.find(ending)
                    simplified = simplified[:idx+1].strip()
                    break
        else:
            # For encoder-decoder models (Models 1-4)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_output_length,
                num_beams=4,
                early_stopping=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
            simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return simplified

def get_cache_filename(cache_key: str) -> Path:
    """Generate cache filename from cache key using hash"""
    import hashlib
    # Use hash to avoid filename issues with special characters
    key_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
    return EVALUATION_CACHE_DIR / f"{key_hash}.json"

def save_evaluation_to_disk(cache_key: str, metrics: dict):
    """Save evaluation results to disk"""
    cache_file = get_cache_filename(cache_key)
    try:
        # Convert numpy arrays and other non-serializable types to Python types
        serializable_metrics = {
            '_cache_key': cache_key,  # Store the cache key in the file
        }
        for key, value in metrics.items():
            if key == 'predictions':
                # Skip predictions as they're large and not needed for display
                continue
            elif key == 'qualitative_samples':
                # Keep qualitative samples but ensure they're serializable
                serializable_metrics[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively handle nested dicts (like length_stats)
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        serializable_metrics[key][k] = {
                            k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2
                            for k2, v2 in v.items()
                        }
                    else:
                        serializable_metrics[key][k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
            else:
                serializable_metrics[key] = value
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Failed to save evaluation cache to disk: {e}")

def load_evaluation_from_disk(cache_key: str) -> Optional[dict]:
    """Load evaluation results from disk"""
    cache_file = get_cache_filename(cache_key)
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Remove the _cache_key from the returned metrics
            metrics = {k: v for k, v in data.items() if k != '_cache_key'}
            return metrics
    except Exception as e:
        st.warning(f"Failed to load evaluation cache from disk: {e}")
        return None

def load_all_disk_cache() -> dict:
    """Load all cached evaluation results from disk"""
    cache = {}
    if not EVALUATION_CACHE_DIR.exists():
        return cache
    
    for cache_file in EVALUATION_CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Get cache key from stored _cache_key field
                cache_key = data.get('_cache_key')
                if cache_key:
                    # Remove the _cache_key from the metrics
                    metrics = {k: v for k, v in data.items() if k != '_cache_key'}
                    cache[cache_key] = metrics
        except Exception as e:
            continue
    
    return cache

def delete_model_4_cache():
    """Delete all cached evaluation results for Model 4 (all 4 combinations)"""
    deleted_count = 0
    deleted_keys = []
    
    if not EVALUATION_CACHE_DIR.exists():
        return deleted_count, deleted_keys
    
    for cache_file in EVALUATION_CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cache_key = data.get('_cache_key', '')
                
                # Check if this is a Model 4 cache file
                if cache_key.startswith("Model 4"):
                    cache_file.unlink()  # Delete the file
                    deleted_count += 1
                    deleted_keys.append(cache_key)
                    
                    # Also remove from session state if it exists
                    if 'evaluation_cache' in st.session_state and cache_key in st.session_state['evaluation_cache']:
                        del st.session_state['evaluation_cache'][cache_key]
        except Exception as e:
            continue
    
    return deleted_count, deleted_keys

def delete_model_2_cache():
    """Delete all cached evaluation results for Model 2 (with and without RAG)"""
    deleted_count = 0
    deleted_keys = []
    
    if not EVALUATION_CACHE_DIR.exists():
        return deleted_count, deleted_keys
    
    for cache_file in EVALUATION_CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cache_key = data.get('_cache_key', '')
                
                # Check if this is a Model 2 cache file (with or without RAG)
                if cache_key.startswith("Model 2"):
                    cache_file.unlink()  # Delete the file
                    deleted_count += 1
                    deleted_keys.append(cache_key)
                    
                    # Also remove from session state if it exists
                    if 'evaluation_cache' in st.session_state and cache_key in st.session_state['evaluation_cache']:
                        del st.session_state['evaluation_cache'][cache_key]
        except Exception as e:
            continue
    
    return deleted_count, deleted_keys

def delete_model_5_cache():
    """Delete all cached evaluation results for Model 5 (all 4 combinations)"""
    deleted_count = 0
    deleted_keys = []
    
    if not EVALUATION_CACHE_DIR.exists():
        return deleted_count, deleted_keys
    
    for cache_file in EVALUATION_CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cache_key = data.get('_cache_key', '')
                
                # Check if this is a Model 5 cache file
                if cache_key.startswith("Model 5"):
                    cache_file.unlink()  # Delete the file
                    deleted_count += 1
                    deleted_keys.append(cache_key)
                    
                    # Also remove from session state if it exists
                    if 'evaluation_cache' in st.session_state and cache_key in st.session_state['evaluation_cache']:
                        del st.session_state['evaluation_cache'][cache_key]
        except Exception as e:
            continue
    
    return deleted_count, deleted_keys

@st.cache_data
def load_test_data(model_name: str):
    """Load test data for the selected model"""
    test_path = TEST_DATA_PATHS.get(model_name)
    if not test_path or not test_path.exists():
        return []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def evaluate_model(
    model,
    tokenizer,
    test_data: List[dict],
    use_rag: bool = False,
    use_contrastive: bool = False,
    is_model_4: bool = False,
    is_model_5: bool = False,
    rag_index=None,
    rag_metadata=None,
    contrastive_encoder=None,
    contrastive_centroids=None,
    max_input_length: int = 512,
    max_output_length: int = 256
) -> dict:
    """Evaluate model on test set and return metrics"""
    # Load metrics
    sari_metric = load_metric("sari")
    bertscore_metric = load_metric("bertscore")
    bleu_metric = load_metric("sacrebleu")
    
    # Prepare data
    source_sentences = [item["legal_sentence"] for item in test_data]
    target_sentences = [item["simplified_sentence"] for item in test_data]
    simplification_types = [item.get("simplification_type") for item in test_data] if (is_model_4 or is_model_5) else [None] * len(test_data)
    
    # If RAG is enabled, batch get all embeddings first (1 API call instead of 100)
    rag_embeddings = None
    if use_rag and rag_index is not None and rag_metadata is not None:
        with st.spinner("Getting embeddings for all queries (batch API call)..."):
            rag_embeddings = batch_get_embeddings(source_sentences)
            if len(rag_embeddings) != len(source_sentences):
                st.warning(f"Expected {len(source_sentences)} embeddings, got {len(rag_embeddings)}. RAG may not work correctly.")
                rag_embeddings = None  # Disable RAG if embeddings don't match
            else:
                st.success(f"✅ Got {len(rag_embeddings)} embeddings in one batch call!")
    
    # Generate predictions
    model.eval()
    predictions = []
    # Track predicted simplification types (for qualitative samples)
    predicted_simplification_types = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = 8
    for i in range(0, len(source_sentences), batch_size):
        batch_sources = source_sentences[i:i+batch_size]
        batch_types = simplification_types[i:i+batch_size] if is_model_4 else [None] * len(batch_sources)
        batch_indices = list(range(i, min(i + batch_size, len(source_sentences))))
        
        # Build prompts for batch
        batch_prompts = []
        for batch_idx, (source, sim_type) in enumerate(zip(batch_sources, batch_types)):
            base_instruction = "የሕግ ቃላትን ለግለሰቦች ለመረዳት ቀላል አማርኛ ውስጥ አቅርብ: "
            
            if is_model_4 or is_model_5:
                type_map = {
                    "vocabulary_simplification": "[የቃላት ማቃለል]",
                    "sentence_splitting": "[የዓረፍተ ነገር መከፋፈል]",
                    "deletion": "[መሻር]",
                    "structure_reordering": "[የዕውቀት ማደራጀት]",
                    "definition_or_expansion": "[ፍቺ ወይም ማስፋፋት]",
                    None: "[አጠቃላይ]"
                }
                # If contrastive is enabled, ALWAYS use contrastive prediction (override dataset values)
                if use_contrastive:
                    if contrastive_encoder and contrastive_centroids:
                        sim_type = predict_simplification_type(source, contrastive_encoder, contrastive_centroids)
                    else:
                        # If contrastive enabled but model not available, use default
                        sim_type = None
                # If contrastive disabled, use dataset value (sim_type from dataset) or None (defaults to "[አጠቃላይ]")
                type_label = type_map.get(sim_type, "[አጠቃላይ]")
                prompt = base_instruction + type_label + " " + source
                
                # Store the simplification_type that was actually used (for qualitative samples)
                predicted_simplification_types.append(sim_type)
            else:
                prompt = base_instruction + source
                # For non-Model-4/5, append None
                predicted_simplification_types.append(None)
            
            # Add RAG context if enabled - use pre-computed embedding
            if use_rag and rag_index is not None and rag_metadata is not None and rag_embeddings is not None:
                # Use the pre-computed embedding for this query
                query_idx = batch_indices[batch_idx]
                query_embedding = rag_embeddings[query_idx]
                rag_context = get_rag_context(source, rag_index, rag_metadata, top_k=2, query_embedding=query_embedding)
                if rag_context:
                    context = "\n".join(rag_context)
                    prompt = f"የሕግ አውድ: {context}\n\n{prompt}"
            
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            if is_model_5:
                # For decoder-only models (Model 5)
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,  # Reduced to limit to single sentence
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.5,  # Increased to prevent repetition
                    no_repeat_ngram_size=4,  # Increased to prevent longer phrase repetition
                    length_penalty=0.6,  # Reduced to encourage shorter outputs
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False  # Use greedy/beam search, not sampling
                )
                # Decode and extract only the generated part (remove input)
                batch_preds_full = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_preds = []
                for pred_full, orig_prompt in zip(batch_preds_full, batch_prompts):
                    if pred_full.startswith(orig_prompt):
                        pred = pred_full[len(orig_prompt):].strip()
                    else:
                        pred = pred_full
                    
                    # Post-process: Extract only the first sentence (stop at first sentence ending)
                    # Amharic sentence endings: ።, ፤, ፥, ፦, and standard punctuation
                    sentence_endings = ['።', '፤', '፥', '፦', '.', '!', '?']
                    for ending in sentence_endings:
                        if ending in pred:
                            # Find first occurrence and take everything up to and including it
                            idx = pred.find(ending)
                            pred = pred[:idx+1].strip()
                            break
                    
                    batch_preds.append(pred)
            else:
                # For encoder-decoder models (Models 1-4)
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_output_length,
                    num_beams=4,
                    early_stopping=False,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
                batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        predictions.extend(batch_preds)
        
        # Update progress
        progress = min((i + batch_size) / len(source_sentences), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing {min(i + batch_size, len(source_sentences))}/{len(source_sentences)} samples...")
    
    progress_bar.empty()
    status_text.empty()
    
    # Compute metrics
    metrics = {}
    
    # BERTScore
    try:
        bertscore_result = bertscore_metric.compute(
            predictions=predictions,
            references=target_sentences,
            lang="am",
            device=device
        )
        metrics["bertscore_f1"] = np.mean(bertscore_result["f1"])
    except Exception as e:
        st.warning(f"BERTScore computation failed: {e}")
        metrics["bertscore_f1"] = 0.0
    
    # SARI
    sari_scores = []
    for source, pred, ref in zip(source_sentences, predictions, target_sentences):
        try:
            sari = sari_metric.compute(
                sources=[source],
                predictions=[pred],
                references=[[ref]]
            )
            sari_scores.append(sari["sari"])
        except Exception as e:
            pass
    metrics["sari"] = np.mean(sari_scores) if sari_scores else 0.0
    
    # BLEU
    try:
        bleu_result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in target_sentences]
        )
        metrics["bleu"] = bleu_result["score"]
    except Exception as e:
        st.warning(f"BLEU computation failed: {e}")
        metrics["bleu"] = 0.0
    
    # Exact Match
    exact_matches = sum(1 for pred, ref in zip(predictions, target_sentences) if pred == ref)
    metrics["exact_match"] = exact_matches / len(predictions) if predictions else 0.0
    metrics["exact_match_count"] = exact_matches
    
    # Length Statistics
    source_lengths = [len(s) for s in source_sentences]
    pred_lengths = [len(p) for p in predictions]
    ref_lengths = [len(r) for r in target_sentences]
    
    metrics["length_stats"] = {
        "source": {
            "mean": np.mean(source_lengths),
            "median": np.median(source_lengths),
            "min": np.min(source_lengths),
            "max": np.max(source_lengths)
        },
        "prediction": {
            "mean": np.mean(pred_lengths),
            "median": np.median(pred_lengths),
            "min": np.min(pred_lengths),
            "max": np.max(pred_lengths)
        },
        "reference": {
            "mean": np.mean(ref_lengths),
            "median": np.median(ref_lengths),
            "min": np.min(ref_lengths),
            "max": np.max(ref_lengths)
        }
    }
    
    length_ratios = [p/s if s > 0 else 0 for p, s in zip(pred_lengths, source_lengths)]
    metrics["length_ratio"] = {
        "mean": np.mean(length_ratios),
        "median": np.median(length_ratios)
    }
    
    # Prepare qualitative samples
    # Use predicted simplification types if contrastive was used, otherwise use dataset values
    if (is_model_4 or is_model_5) and use_contrastive and len(predicted_simplification_types) == len(source_sentences):
        types_to_use = predicted_simplification_types
    else:
        types_to_use = simplification_types if (is_model_4 or is_model_5) else [None] * len(source_sentences)
    
    qualitative_samples = []
    for i, (source, pred, ref, sim_type) in enumerate(zip(
        source_sentences, predictions, target_sentences, types_to_use
    )):
        qualitative_samples.append({
            "legal_sentence": source,
            "simplified_sentence": ref,
            "predicted": pred,
            "simplification_type": sim_type if ((is_model_4 or is_model_5) and sim_type) else None
        })
    
    metrics["qualitative_samples"] = qualitative_samples
    metrics["predictions"] = predictions
    
    return metrics

# Main app
st.title("📜 Amharic Legal Text Simplification System")
st.markdown("---")

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    
    # Model selection
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(ADAPTER_PATHS.keys()),
        index=0
    )
    
    # RAG selection
    use_rag = st.selectbox(
        "RAG Intervention",
        options=["Without RAG", "With RAG"],
        index=0
    )
    use_rag = use_rag == "With RAG"
    
    # Contrastive selection (only for Model 4 and Model 5)
    use_contrastive = False
    if "Model 4" in selected_model_name or "Model 5" in selected_model_name:
        contrastive_option = st.selectbox(
            "Contrastive Learning",
            options=["Without Contrastive", "With Contrastive"],
            index=0
        )
        use_contrastive = contrastive_option == "With Contrastive"
    
    st.markdown("---")
    
    # Test and Compare button
    if st.button("🧪 Test and Compare", type="primary", use_container_width=True):
        st.session_state['run_evaluation'] = True
        st.session_state['selected_model'] = selected_model_name
        st.session_state['use_rag'] = use_rag
        st.session_state['use_contrastive'] = use_contrastive
        st.session_state['switch_to_evaluation'] = True  # Flag to switch to evaluation tab
        st.success("✅ Evaluation started! Switch to the 'Evaluation Results' tab to view results.")
    
    st.markdown("---")
    st.header("Cache Management")
    
    # Delete Model 2 cache button
    if st.button("🗑️ Delete Model 2 Cache", use_container_width=True):
        deleted_count, deleted_keys = delete_model_2_cache()
        if deleted_count > 0:
            st.success(f"✅ Deleted {deleted_count} cache file(s) for Model 2")
            st.write("Deleted keys:", deleted_keys)
        else:
            st.info("No Model 2 cache files found to delete")
    
    # Delete Model 5 cache button
    if st.button("🗑️ Delete Model 5 Cache", use_container_width=True):
        deleted_count, deleted_keys = delete_model_5_cache()
        if deleted_count > 0:
            st.success(f"✅ Deleted {deleted_count} cache file(s) for Model 5")
            st.write("Deleted keys:", deleted_keys)
        else:
            st.info("No Model 5 cache files found to delete")

# Initialize evaluation cache in session state
# Load from disk cache on startup
if 'evaluation_cache' not in st.session_state:
    st.session_state['evaluation_cache'] = load_all_disk_cache()
    if st.session_state['evaluation_cache']:
        st.info(f"📁 Loaded {len(st.session_state['evaluation_cache'])} cached evaluation results from disk.")

# Create tabs for main content
tab1, tab2 = st.tabs(["✨ Simplify Text", "🧪 Evaluation Results"])

# Tab 1: Simplification Interface (clean, no evaluation metrics)
with tab1:
    st.header("Simplification Interface")
    st.markdown("Enter a legal sentence in Amharic to get a simplified version using the selected model configuration.")
    
    # Input field
    input_text = st.text_area(
        "Enter Legal Sentence (Amharic)",
        height=150,
        placeholder="የሕግ ቃላትን እዚህ ያስገቡ...",
        key="simplify_input"
    )
    
    # Simplify button
    if st.button("Simplify", type="primary", use_container_width=True, key="simplify_button"):
        if not input_text.strip():
            st.error("Please enter a legal sentence to simplify.")
        else:
            with st.spinner("Simplifying..."):
                try:
                    # Load base model
                    base_model, tokenizer = load_base_model(selected_model_name)
                    
                    # Load adapter
                    adapter_path = ADAPTER_PATHS[selected_model_name]
                    model = load_adapter(base_model, adapter_path)
                    
                    # Get simplification type if using contrastive (Model 4 or Model 5)
                    simplification_type = None
                    if use_contrastive and ("Model 4" in selected_model_name or "Model 5" in selected_model_name):
                        contrastive_result = load_contrastive_selector()
                        if contrastive_result:
                            encoder, centroids = contrastive_result
                            simplification_type = predict_simplification_type(input_text, encoder, centroids)
                            st.info(f"Predicted simplification type: **{simplification_type}**")
                    
                    # Get RAG context if enabled
                    rag_context = None
                    if use_rag:
                        index, metadata = load_rag_system()
                        if index is not None and metadata is not None:
                            rag_context = get_rag_context(input_text, index, metadata)
                            if rag_context:
                                st.info(f"Retrieved {len(rag_context)} relevant legal contexts")
                    
                    # For Model 4 and Model 5, always pass simplification_type (use default if contrastive not enabled)
                    if ("Model 4" in selected_model_name or "Model 5" in selected_model_name) and not simplification_type:
                        simplification_type = None  # Will use default "[አጠቃላይ]" in simplify_text
                    
                    # Simplify
                    simplified = simplify_text(
                        input_text,
                        model,
                        tokenizer,
                        simplification_type=simplification_type,
                        rag_context=rag_context,
                        is_model_4=("Model 4" in selected_model_name),
                        is_model_5=("Model 5" in selected_model_name)
                    )
                    
                    # Display result
                    st.success("Simplification Complete!")
                    st.subheader("Simplified Text")
                    st.write(simplified)
                    
                    # Show comparison
                    with st.expander("View Comparison"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Original:**")
                            st.write(input_text)
                        with col_b:
                            st.write("**Simplified:**")
                            st.write(simplified)
                    
                except Exception as e:
                    st.error(f"Error during simplification: {str(e)}")
                    st.exception(e)

# Tab 2: Evaluation Results
with tab2:
    st.header("🧪 Model Evaluation Results")
    
    # Show message if evaluation was just triggered
    if st.session_state.get('switch_to_evaluation', False):
        st.info("🔄 Evaluation is running or has completed. Results will appear below.")
        st.session_state['switch_to_evaluation'] = False  # Reset flag
    
    # Check if evaluation should run
    if 'run_evaluation' in st.session_state and st.session_state['run_evaluation']:
        eval_selected_model = st.session_state.get('selected_model', selected_model_name)
        eval_use_rag = st.session_state.get('use_rag', use_rag)
        eval_use_contrastive = st.session_state.get('use_contrastive', use_contrastive)
        
        # Create cache key
        cache_key = f"{eval_selected_model}_{eval_use_rag}_{eval_use_contrastive}"
        
        # Check disk cache first (persistent across restarts)
        metrics = load_evaluation_from_disk(cache_key)
        if metrics:
            # Load from disk and also store in session state for faster access
            st.session_state['evaluation_cache'][cache_key] = metrics
            st.info("📁 Using cached evaluation results from disk. Click 'Test and Compare' again to re-evaluate.")
        # Check session state cache (in-memory, faster)
        elif cache_key in st.session_state['evaluation_cache']:
            # Just show cached results immediately - no re-evaluation!
            metrics = st.session_state['evaluation_cache'][cache_key]
        else:
            # Only run evaluation if cache doesn't exist
            # Load test data
            test_data = load_test_data(eval_selected_model)
            if not test_data:
                st.error(f"Test data not found for {eval_selected_model}")
                metrics = None
            else:
                st.info(f"Evaluating on {len(test_data)} test samples...")
                
                # Load models and systems
                with st.spinner("Loading models and systems..."):
                    base_model, tokenizer = load_base_model(eval_selected_model)
                    adapter_path = ADAPTER_PATHS[eval_selected_model]
                    model = load_adapter(base_model, adapter_path)
                    
                    rag_index, rag_metadata = None, None
                    if eval_use_rag:
                        rag_index, rag_metadata = load_rag_system()
                        if rag_index is None:
                            st.warning("RAG system not available, running without RAG")
                            eval_use_rag = False
                    
                    contrastive_encoder, contrastive_centroids = None, None
                    if eval_use_contrastive and ("Model 4" in eval_selected_model or "Model 5" in eval_selected_model):
                        contrastive_result = load_contrastive_selector()
                        if contrastive_result:
                            contrastive_encoder, contrastive_centroids = contrastive_result
                        else:
                            st.warning("Contrastive selector not available, running without contrastive")
                            eval_use_contrastive = False
                
                # Run evaluation for the selected configuration only
                is_model_4 = "Model 4" in eval_selected_model
                is_model_5 = "Model 5" in eval_selected_model
                
                with st.spinner("Evaluating model..."):
                    metrics = evaluate_model(
                        model, tokenizer, test_data,
                        use_rag=eval_use_rag,
                        use_contrastive=eval_use_contrastive if (is_model_4 or is_model_5) else False,
                        is_model_4=is_model_4,
                        is_model_5=is_model_5,
                        rag_index=rag_index if eval_use_rag else None,
                        rag_metadata=rag_metadata if eval_use_rag else None,
                        contrastive_encoder=contrastive_encoder if eval_use_contrastive else None,
                        contrastive_centroids=contrastive_centroids if eval_use_contrastive else None
                    )
                
                # Cache the results in session state (in-memory)
                st.session_state['evaluation_cache'][cache_key] = metrics
                # Also save to disk (persistent across restarts)
                save_evaluation_to_disk(cache_key, metrics)
                st.success("✅ Evaluation complete! Results are cached in memory and saved to disk.")
        
        # Reset evaluation flag
        st.session_state['run_evaluation'] = False
    
    # Display results for current selection (cached or from evaluation)
    current_cache_key = f"{selected_model_name}_{use_rag}_{use_contrastive}"
    
    # Check if we have cached results for current selection
    metrics = load_evaluation_from_disk(current_cache_key)
    if metrics:
        # Load from disk and store in session state
        if 'evaluation_cache' not in st.session_state:
            st.session_state['evaluation_cache'] = {}
        st.session_state['evaluation_cache'][current_cache_key] = metrics
    elif 'evaluation_cache' in st.session_state and current_cache_key in st.session_state['evaluation_cache']:
        metrics = st.session_state['evaluation_cache'][current_cache_key]
    
    if metrics:
        # Show configuration
        rag_status = "With RAG" if use_rag else "Without RAG"
        contrastive_status = f", With Contrastive" if (use_contrastive and ("Model 4" in selected_model_name or "Model 5" in selected_model_name)) else ""
        st.subheader(f"{selected_model_name} - {rag_status}{contrastive_status}")
        
        # Display quantitative metrics
        st.subheader("Quantitative Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BERTScore F1", f"{metrics['bertscore_f1']:.4f}")
        with col2:
            st.metric("SARI", f"{metrics['sari']:.4f}")
        with col3:
            st.metric("BLEU", f"{metrics['bleu']:.4f}")
        with col4:
            test_data = load_test_data(selected_model_name)
            st.metric("Exact Match", f"{metrics['exact_match']:.4f} ({metrics['exact_match_count']}/{len(test_data)})")
        
        # Length statistics
        st.subheader("Length Statistics")
        len_stats = metrics['length_stats']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Source (Original)**")
            st.write(f"Mean: {len_stats['source']['mean']:.1f} chars")
            st.write(f"Median: {len_stats['source']['median']:.1f} chars")
            st.write(f"Range: {len_stats['source']['min']}-{len_stats['source']['max']} chars")
        with col2:
            st.write("**Prediction (Model)**")
            st.write(f"Mean: {len_stats['prediction']['mean']:.1f} chars")
            st.write(f"Median: {len_stats['prediction']['median']:.1f} chars")
            st.write(f"Range: {len_stats['prediction']['min']}-{len_stats['prediction']['max']} chars")
        with col3:
            st.write("**Reference (Target)**")
            st.write(f"Mean: {len_stats['reference']['mean']:.1f} chars")
            st.write(f"Median: {len_stats['reference']['median']:.1f} chars")
            st.write(f"Range: {len_stats['reference']['min']}-{len_stats['reference']['max']} chars")
        
        st.write(f"**Length Ratio (Prediction/Source):** Mean: {metrics['length_ratio']['mean']:.3f}, Median: {metrics['length_ratio']['median']:.3f}")
        
        # Qualitative samples
        st.subheader("Qualitative Samples")
        num_samples = st.slider("Number of samples to display", 1, min(20, len(metrics['qualitative_samples'])), 3, key="samples_display")
        
        for i, sample in enumerate(metrics['qualitative_samples'][:num_samples], 1):
            with st.expander(f"Sample {i}"):
                st.write("**Legal Sentence:**", sample['legal_sentence'])
                if sample.get('simplification_type'):
                    st.write("**Simplification Type:**", sample['simplification_type'])
                st.write("**Reference Simplified:**", sample['simplified_sentence'])
                st.write("**Model Prediction:**", sample['predicted'])
        
        st.info("💡 Results are cached. Click 'Test and Compare' to view these results or evaluate a different configuration.")
    else:
        # Show placeholder or instruction
        st.info("👆 Select a model configuration and click 'Test and Compare' to see evaluation results.")
        
        # Optionally show training details from EVALUATION_RESULTS
        if selected_model_name in EVALUATION_RESULTS:
            results = EVALUATION_RESULTS[selected_model_name]
            st.subheader("Training Details")
            st.write(f"**Dataset:** {results.get('Dataset', 'N/A')}")
            st.write(f"**Configuration:** {results.get('Config', 'N/A')}")

# Footer
st.markdown("---")
st.markdown("""
### System Components:
- **5 Fine-tuned Adapters**: Trained on different datasets and configurations
- **RAG System**: Retrieval-augmented generation for legal context
- **Contrastive Strategy Selector**: Predicts simplification type for Model 4 and Model 5
""")

