# Amharic Legal Text Simplification - Streamlit App

## Overview

This Streamlit application provides a unified interface to test and compare all 4 fine-tuned models for Amharic legal text simplification, with optional RAG intervention and contrastive learning for Model 4.

## Features

- **4 Model Options**: Select from 4 different fine-tuned adapters
  - Model 1: Trained on `final.json` with old config (r=16, alpha=32)
  - Model 2: Trained on `final.json` with updated config (r=32, alpha=64)
  - Model 3: Trained on `regen.json` without simplification_type
  - Model 4: Trained on `regen.json` with simplification_type conditioning

- **RAG Intervention**: Toggle RAG (Retrieval-Augmented Generation) on/off for all models
- **Contrastive Learning**: Toggle contrastive strategy selector on/off (Model 4 only)
- **Evaluation Results**: View metrics and qualitative samples for each model
- **Interactive Simplification**: Input legal sentences and get simplified output

## Installation

1. Install required packages:
```bash
pip install -r requirements_streamlit.txt
```

2. Ensure all models are downloaded:
   - Base model: `models/afri-byt5-base/`
   - Adapters: `models/afribyt5-legal-simplification-final*/`
   - Contrastive selector: `models/contrastive_strategy_selector/`
   - RAG system: `rag_pipeline/4_vector_db/`

3. Set up Gemini API key (for RAG):
   - Place your API key in `.gemini_api_key` file in the project root

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select Model**: Choose one of the 4 models from the dropdown
2. **Choose RAG**: Select "With RAG" or "Without RAG"
3. **Contrastive (Model 4 only)**: If Model 4 is selected, choose "With Contrastive" or "Without Contrastive"
4. **View Results**: Evaluation metrics and qualitative samples are displayed automatically
5. **Simplify Text**: Enter a legal sentence in Amharic and click "Simplify"

## Model Combinations

The app supports the following combinations:

1. **Model 1-4 (bare)**: Without RAG, without contrastive
2. **Model 1-4 (with RAG)**: With RAG, without contrastive
3. **Model 4 (with contrastive)**: Without RAG, with contrastive
4. **Model 4 (with contrastive + RAG)**: With RAG, with contrastive

## Updating Evaluation Results

The evaluation results displayed in the app are currently placeholders. To update them with actual results from your training notebooks:

1. Extract evaluation metrics (BERTScore, SARI) from notebook outputs
2. Extract qualitative samples from notebook outputs
3. Update the `EVALUATION_RESULTS` dictionary in `streamlit_app.py`

## File Structure

```
.
├── streamlit_app.py              # Main Streamlit application
├── requirements_streamlit.txt    # Python dependencies
├── models/                       # Model directory
│   ├── afri-byt5-base/          # Base model
│   ├── afribyt5-legal-simplification-final*/  # Adapters
│   └── contrastive_strategy_selector/  # Contrastive model
├── rag_pipeline/                # RAG system files
│   └── 4_vector_db/             # FAISS index and metadata
└── .gemini_api_key              # Gemini API key (create this)

```

## Troubleshooting

- **Model not loading**: Ensure models are downloaded and paths are correct
- **RAG not working**: Check that `.gemini_api_key` exists and contains valid API key
- **Contrastive not working**: Ensure `models/contrastive_strategy_selector/` exists with `centroids.pkl`
- **Memory issues**: Models are cached, but initial load may require significant RAM/VRAM

## Notes

- Models are loaded lazily and cached for performance
- RAG requires internet connection for Gemini API calls
- First run may be slow due to model loading

