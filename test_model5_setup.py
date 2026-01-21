"""
Quick test script to verify Model 5 setup before running Streamlit app
"""

from pathlib import Path
import sys

WORKDIR = Path(__file__).parent
MODELS_DIR = WORKDIR / "models"

print("=" * 60)
print("Testing Model 5 Setup")
print("=" * 60)
print()

# Check adapter folder
adapter_path = MODELS_DIR / "llama-400m-legal-simplification5"
print(f"1. Checking adapter folder: {adapter_path}")
if adapter_path.exists():
    adapter_config = adapter_path / "adapter_config.json"
    adapter_model = adapter_path / "adapter_model.safetensors"
    if adapter_config.exists() and adapter_model.exists():
        print("   ✅ Adapter folder found with required files")
    else:
        print("   ⚠️  Adapter folder exists but missing some files")
        sys.exit(1)
else:
    print("   ❌ Adapter folder not found!")
    sys.exit(1)

print()

# Check base model
base_model_path = MODELS_DIR / "models--rasyosef--Llama-3.2-400M-Amharic-Instruct"
print(f"2. Checking base model: {base_model_path}")
if base_model_path.exists():
    snapshots = base_model_path / "snapshots"
    if snapshots.exists():
        snapshot_dirs = list(snapshots.iterdir())
        if snapshot_dirs:
            print(f"   ✅ Base model found with {len(snapshot_dirs)} snapshot(s)")
        else:
            print("   ⚠️  Base model folder exists but no snapshots found")
            sys.exit(1)
    else:
        print("   ⚠️  Base model folder exists but no snapshots directory")
        sys.exit(1)
else:
    print("   ❌ Base model not found!")
    print("   💡 Run: python download_llama_model.py")
    sys.exit(1)

print()

# Check test data
test_data_path = WORKDIR / "Dataset" / "final_dataset" / "regen_test.json"
print(f"3. Checking test data: {test_data_path}")
if test_data_path.exists():
    print("   ✅ Test data found")
else:
    print("   ⚠️  Test data not found (may still work)")

print()

# Check Streamlit app
streamlit_app = WORKDIR / "streamlit_app.py"
print(f"4. Checking Streamlit app: {streamlit_app}")
if streamlit_app.exists():
    print("   ✅ Streamlit app found")
else:
    print("   ❌ Streamlit app not found!")
    sys.exit(1)

print()
print("=" * 60)
print("✅ All checks passed! Model 5 is ready to use.")
print("=" * 60)
print()
print("To run the Streamlit app:")
print("  streamlit run streamlit_app.py")
print()
print("Or if using venv:")
print("  source venv/bin/activate")
print("  streamlit run streamlit_app.py")
print()



