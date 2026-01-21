"""
Download LLaMA 3.2 400M Amharic Instruct model to local models directory
This script downloads the model and tokenizer to be used by the Streamlit app
"""

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Constants
WORKDIR = Path(__file__).parent
MODELS_DIR = WORKDIR / "models"
MODEL_NAME = "rasyosef/Llama-3.2-400M-Amharic-Instruct"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

print(f"Downloading {MODEL_NAME}...")
print(f"Target directory: {MODELS_DIR}")
print(f"This may take a while depending on your internet connection...")
print()

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=str(MODELS_DIR),
    trust_remote_code=True
)
print("✅ Tokenizer downloaded successfully!")
print()

# Download model
print("Downloading model (this is the large file, ~800MB)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=str(MODELS_DIR),
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("✅ Model downloaded successfully!")
print()

# Verify the download location
print("Verifying download location...")
expected_path = MODELS_DIR / "models--rasyosef--Llama-3.2-400M-Amharic-Instruct"
if expected_path.exists():
    print(f"✅ Model cached at: {expected_path}")
    snapshots = list((expected_path / "snapshots").iterdir())
    if snapshots:
        print(f"✅ Found snapshot: {snapshots[0].name}")
else:
    print(f"⚠️  Expected path not found, but model may be cached elsewhere in {MODELS_DIR}")

print()
print("=" * 60)
print("Download complete!")
print(f"Model is ready to use in the Streamlit app.")
print(f"Cache location: {MODELS_DIR}")
print("=" * 60)

