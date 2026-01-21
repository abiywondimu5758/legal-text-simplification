# Local Evaluation Setup for Trained Models

## Overview
All 4 models have been trained and saved in the `models/` folder. To evaluate them locally, the notebooks need to be updated to:

1. **Load trained models** from `models/` folder instead of training
2. **Use local dataset paths** instead of `/content/` or `/kaggle/input/`
3. **Run evaluation** with all metrics (SARI, BERTScore, BLEU, Exact Match, Length Statistics)

## Model Locations

- **Model 1** (final.json, basic): `models/afribyt5-legal-simplification-final/`
- **Model 2** (final.json, improved): `models/afribyt5-legal-simplification-final2/`
- **Model 3** (regen.json, with instruction): `models/afribyt5-legal-simplification-final3/`
- **Model 4** (regen.json, with simplification_type): `models/afribyt5-legal-simplification-final4/`

## Dataset Locations

- **final.json splits**: 
  - `Dataset/final_dataset/final_train.json`
  - `Dataset/final_dataset/final_val.json`
  - `Dataset/final_dataset/final_test.json`

- **regen.json splits**:
  - `Dataset/final_dataset/regen_train.json`
  - `Dataset/final_dataset/regen_val.json`
  - `Dataset/final_dataset/regen_test.json`

## Base Model Location

- Base model: `models/afri-byt5-base/` (or use `masakhane/afri-byt5-base` to download if needed)

## What Needs to be Updated

1. **Dataset loading paths**: Change from `/content/` or `/kaggle/input/` to `Dataset/final_dataset/`
2. **Model loading**: Add cells to load base model + adapter using `PeftModel.from_pretrained()`
3. **Evaluation cells**: Ensure all metrics are computed (BLEU, Exact Match, Length Statistics already added ✓)

## Evaluation Metrics Included

✅ SARI (primary)
✅ BERTScore F1 (secondary)
✅ BLEU Score
✅ Exact Match Rate
✅ Length Statistics (mean, median, min, max, ratio)







