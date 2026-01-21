#!/usr/bin/env python3
"""
Script to create Colab and Kaggle versions of afribyt5 notebooks with validation and visualization.
This script reads the original notebooks and creates updated versions.
"""

import json
import sys
from pathlib import Path

def add_visualization_and_fix_notebook(notebook_path, is_kaggle=False):
    """Add visualization section and fix paths for Colab or Kaggle"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find cells to modify
    cells = nb['cells']
    
    # 1. Add matplotlib import to first import cell
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and 'import' in ''.join(cell['source']):
            if 'matplotlib' not in ''.join(cell['source']):
                source = cell['source']
                # Find where to insert matplotlib
                for j, line in enumerate(source):
                    if 'import torch' in line:
                        source.insert(j+1, 'import matplotlib.pyplot as plt\n')
                        break
                cell['source'] = source
            break
    
    # 2. Fix data paths
    data_path_colab = '/content/'
    data_path_kaggle = '/kaggle/input/legal-sentence-simplifier/'
    
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'final_train.json' in source or 'final_val.json' in source or 'final_test.json' in source:
                if is_kaggle:
                    cell['source'] = [line.replace(data_path_colab, data_path_kaggle) 
                                    if 'final_' in line else line 
                                    for line in cell['source']]
                else:
                    cell['source'] = [line.replace(data_path_kaggle, data_path_colab) 
                                    if 'final_' in line else line 
                                    for line in cell['source']]
    
    # 3. Fix output paths
    output_path_colab = './'
    output_path_kaggle = '/kaggle/working/'
    
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'output_dir' in source or 'save_model' in source or 'save_pretrained' in source:
                if is_kaggle:
                    cell['source'] = [line.replace('./', output_path_kaggle) 
                                    if ('output_dir' in line or 'save' in line) else line 
                                    for line in cell['source']]
                else:
                    cell['source'] = [line.replace(output_path_kaggle, output_path_colab) 
                                    if ('output_dir' in line or 'save' in line) else line 
                                    for line in cell['source']]
    
    return nb

if __name__ == '__main__':
    print("This script would process notebooks, but manual editing is more reliable.")
    print("Please use the edit_notebook tool for each notebook individually.")
