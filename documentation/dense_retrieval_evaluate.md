# Encode & index & retrieve a trained dense retrieval model (BERT_DOT)

Our main script for the 3 phases of dense retrieval is ``dense_retrieval.py`` 🔎

``dense_retrieval.py`` allows you to conduct 3 phases of dense retrieval: encoding, indexing, search (& evaluation) in batch form (not really meant for production use of a search engine):

- Needs a trained dense retrieval model (via [matchmaker/train.py](../matchmaker/train.py) or the HuggingFace modelhub)
- Measures efficiency & effectiveness on 1 collection + multiple query sets (start a new experiment for another collection)
- Allows to start new experiment from each of the 3 steps via modes:

|**Mode**       | **Config requirement** |
| ------------- |:-------------          |
|1) encode+index+search | *trained_model* folder path or huggingface model name |
|2) index+search        | *continue_folder* folder path pointing to an experiment started with 1) |
|3) search              | *continue_folder* folder path pointing to an experiment started with 2) |

- We can do a lot of hyperparameter studies starting from each step, or just run through a full pass once

- We support onnx-runtime inference for encoding (only useful when all optimizations & fp16 mode is used)
    - To use it set: ``onnx_use_inference: True`` (the pytorch model gets converted on the fly, which leads to higher GPU memory requirements)
    - We observe large efficiency gains: 5,500 passages / second encoding speed for DistilBERT (up from 3,300 when using PyTorch), single query encoding latency takes < 1ms (down from ~7ms on PyTorch) on a single TITAN RTX.
    - Top speed measured on a single A40 (40GB, encoding batch size 3000) = **6,900 MSMARCOv1 passages per second**
    - The effect on effectiveness is mostly inside a tenth of a point (but some more experiments are necessary)

- Every start of ``dense_retrieval.py`` creates a new experiment folder (copies the current code & config) and saves the results 

## CLI of dense_retrieval.py

To use ``dense_retrieval.py`` you need to set the mode as the first parameter, as well as a run-name, config path and optional config-overwrites.

````
python matchmaker/dense_retrieval.py <mode> --run-name <the_experiment_name> --config <the_experiment_name> --config-overwrites "<optional override1>,<optional override2> ..."
````

For an example config see the *[config/dense_retrieval/minimal-usage-example.yaml](../config/dense_retrieval/minimal-usage-example.yaml)*.


## Mode 1) A full run via encode+index+search

Example full encode & index & search command (on the first 3 GPUs of the system):
````
CUDA_VISIBLE_DEVICES=0,1,2 python matchmaker/dense_retrieval.py encode+index+search --run-name <the_experiment_name> --config config/dense_retrieval/tr-msmarco.yaml
````

The script saves all encoded collection vectors & mapping information; if a non-full index is used (such as hnsw or ivf) the Faiss-index is saved as well. This allows us to re-use the work in the other modes.

## Mode 2) Run ablation studies on an encoded collection in index+search

Needs a completed experiment from 1) with all collection vectors saved. 

A simple bash loop for hnsw parameter sweeping, for the sample parameters we need to do index+search, (but we could also could only run the "search" mode if we only need to change query time params):
````
faiss_param=(64 128 256 512 1024)
for i in ${faiss_param[@]}; do
python matchmaker/dense_retrieval.py index+search --run-name tasb_256_hnsw${i} --config config/dense_retrieval/tr-msmarco.yaml \
--config-overwrites "continue_folder: <path_to_experiment_folder_of_1)>,faiss_index_type: hnsw,faiss_hnsw_graph_neighbors: ${i},faiss_hnsw_efConstruction: ${i},faiss_hnsw_efSearch: ${i}"
done
````

## Evaluate / Run our published models from the HuggingFace model hub 🤗

Currently we support loading our own huggingface models (https://huggingface.co/sebastian-hofstaetter), but configs can be easily added to support more.

The models currently tested are:

- BERT_Dot (Margin-MSE T2)  from: https://github.com/sebastian-hofstaetter/neural-ranking-kd
- BERT_Dot (TAS-Balanced) from: https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval

In the config file set ``trained_model: sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco`` and ``dense_retrieval.py`` automatically looks up our local config in the *config/huggingface_modelhub* folder & downloads the necessary weight files and loads them. 
