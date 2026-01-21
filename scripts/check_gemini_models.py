# check_gemini_models.py
import google.generativeai as genai

# Set your API key
API_KEY = "REPLACE WITH YOUR API KEY HERE"
genai.configure(api_key=API_KEY)

# List all available models
print("Available Gemini models:")
print("=" * 50)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✓ {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Description: {model.description}")
        print()