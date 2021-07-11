# Use matchmaker to train the IDCM model

## Prepare data

We require an MSMARCO tsv data format for all text files, TREC qrel format. The config is mainly set via .yaml files (can be split into multiple files for easy experiment mgmt) and overridden via cli. 

The data-prep has to be repeated for passage & document collections.

- **Train** triples in tsv format
  - The dataset config file should set the config as glob-able path: ``train_tsv: "path/to/train.tsv"``


- **Validation**
  - Matchmaker uses two different validation sets: 1 for continuous validation ``validation_cont`` (cadence via ``validate_every_n_batches``) and for a final validation round a second set can be used ``validation_end`` (with greater depth / more queries f.e. or just re-configure the same set)
    
    Here, we need the "``q_id    doc_id    q_text    doc_text``" tsv text file as well as a bm25 candidate file (for re-ranking depth evaluation and analysis) standard Anserini/TREC result format.

    Example:
    ````
    validation_cont:
      binarization_point: 1
      candidate_set_from_to: [5,100]
      candidate_set_path: /path/to/plain_bm25_result_top100.txt
      qrels: /path/to/qrels/qrels.dev.txt
      save_only_best: true
      tsv: /path/to/validation_tuples_text.tsv
    ````

- **Test**
   - Same definition as the validation, but only computed at the end of training once
  
      Example:
      ````
      test:
        trec2019_rerank_100:
          tsv: "/path/to/msmarco-doctest2019-top100.tsv"
          qrels: "/path/to/qrels/trec2019-qrels-docs.txt"
          binarization_point: 2
          save_secondary_output: True
      ````

   - If no qrel info available (use ``leaderboard:``)

**Important config settings**
````
max_doc_length: 2000
max_query_length: 30

batch_size_train: 32
batch_size_eval: 124

loss: "ranknet"
use_fp16: True
````
-> but those are not the only ones that need to be set, in general all setting keys that are in configs/defaults.yaml should be loaded every time to ensure the code working properly and not breaking halfway through training


## Multi-stage training

1. **BERT Passage Training** 

We train a bert_cat model on passage data without splitting. Relevant config:
````
token_embedder_type: "bert_cat" 
model: "bert_cat" 
````

Example train command (cd in ir-project-matchmaker):
````
CUDA_VISIBLE_DEVICES=0,1,2,3 python matchmaker/train.py --config-file matchmaker/configs/defaults.yaml matchmaker/configs/datasets/<your dataset here>.yaml matchmaker/configs/models/idcm.yaml --run-name idcm_passage_train
````


The matchmaker library automatically saves the best model checkpoint (if a higher ``validation_metric`` validation value is reached)

2. **Full BERT Document Training** 

We now train the BERT part of the IDCM model on document data, but we warmstart the model with the bert_cat checkpoint from 1. Relevant config:

````
token_embedder_type: "bert_dot" 
model: "IDCM" 
warmstart_model_path: "path/to/model_from_1/best-model.pytorch-state-dict"
````

Example train command (cd in ir-project-matchmaker), not the sample_n is set to -1 to deactivate it (could also be done in the config files)
````
CUDA_VISIBLE_DEVICES=0,1,2,3 python matchmaker/train.py --config-file matchmaker/configs/defaults.yaml matchmaker/configs/datasets/<your dataset here>.yaml matchmaker/configs/models/idcm.yaml --config-overwrites "idcm_sample_n: -1" --run-name idcm_full_doc_train
````

3. **Selection Component Document Training** 

We now train the selection part of the IDCM model on document data. Relevant config:

````
token_embedder_type: "bert_dot" 
model: "IDCM" 
warmstart_model_path: "path/to/model_from_2/best-model.pytorch-state-dict"

# this enables the second in-document loss, badly named sparsity at the moment, once idcm_sample_n is > 0, we disable the ranknet loss by detaching the bert output from the gradient taping
minimize_sparsity: True
sparsity_loss_lambda_factor: 1
sparsity_log_path: "sparsity-info.tsv"
sparsity_reanimate: False
````

Optionally we can cache BERT scores at this point. The cache is independent of idcm_sample_n and other sampling attributes, but needs the exact same batchsizes, and input files to work properly (as there are no ids to match, we replay the cache)  With the first training we save them, and can then utilize the built cache via:

````
submodel_train_cache_path: "/path-to-cache/bertpatch_max2k_cache_train_bs32"
submodel_validation_cache_path: "/path-to-cache/bertpatch_max2k_cache_val_bs124"

# set False for the first writing session
submodel_train_cache_readonly: True
submodel_validation_cache_readonly: true
````

Example train command (cd in ir-project-matchmaker), sample_n is now set to > 0
````
CUDA_VISIBLE_DEVICES=0,1,2,3 python matchmaker/train.py --config-file matchmaker/configs/defaults.yaml matchmaker/configs/datasets/<your dataset here>.yaml matchmaker/configs/models/idcm.yaml --config-overwrites "idcm_sample_n: 3" --run-name idcm_full_doc_train
````
