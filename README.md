# CorrEvator: Evaluating Patch Correctness in Automated Program Repair from Online Community Discussions

CorrEvator (Correct Patches Evaluator) is a tool used for automatically evaluating the correctness of program patches, mainly applied in the field of automatic program repair. Its core objective is to determine whether a patch truly fixes a bug by analyzing the semantic and structural correlations between error reports and patch descriptions.

## Python version

    python 3.9

## Install pytorch and pyg 

For PyTorch and PyG, the following are the direct installation commands that can be used (it is recommended to use the conda environment).

    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

## Word Segmentation

    pip install spacy (or conda install -c conda-forge spacy)
    python -m spacy download en_core_web_sm (This file has been provided.)

## Dataset

The dataset is located in the `bugreport_patch.txt` file within the `data` directory.

Please `run pre_process.py` and `run get_graph.py` to construct the data into a graph structure.

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

## Model

Here is a model that we have trained for testing. You can also train your own model following the above steps.

If you want to run the model we provide, please run these file:

```
pre_process.py -> get_graph.py -> get_metrics.py
```

If you want to train  your own model, please run these file:

```
pre_process.py -> get_graph.py -> train.py -> get_metrics.py
```

