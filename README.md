# CorrEvator
## Python version

```
python 3.9
```

## Installation of PyTorch and PyG

```
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

## Word Segmentation

```
pip install spacy or conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
```

## File Descriptions
```
requirements.txt:   Required packages
pre_process.py:     Data preprocessing and dataset splitting
get_graph.py:       Graph construction
gmn.py:             Model code
train.py:           Train the model
get_metrics.py:     Evaluate the model
run.py:             Run all scripts with one command
```
