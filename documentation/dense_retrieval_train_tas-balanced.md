# Train a Dense Retriever (BERT_DOT) with TAS-Balanced & Dual-Supervision

This guide builds up on: [dense_retrieval_train.md](dense_retrieval_train.md)

We again use a BERT_DOT with shared encoder: **bert_dot**

## TAS-Balanced

For the dataset config, see the *config/train/data/example-minimal-tasb-dataset.yaml*.

The config for TAS-Balanced input files:
````yaml
dynamic_query_file: "/path/to/train/queries.train.tsv"
dynamic_collection_file: "/path/to/train/collection.tsv"
dynamic_pairs_with_teacher_scores: "/path/to/train/T2-train-ids.tsv" # output of matchmaker/distillation/teacher_textscore_to_ids.py (from single model pairwise scores or the ensemble)
dynamic_query_cluster_file: "/path/to/train/train_clusters.tsv"      # generate it with matchmaker/distillation/query_clusterer.py (and a baseline dense retrieval model)
````

Example train command (cd in ir-project-matchmaker):
````
python matchmaker/train.py --config-file config/train/defaults.yaml config/train/data/<your dataset here>.yaml config/train/models/bert_dot.yaml config/train/modes/tas_balanced.yaml --run-name your_experiment_name
````

The dynamic_teacher will run on the same GPU as the main training (if you only have 1 GPU available) if you have more than 1 GPU available, the dynamic teacher will exclusively take the last GPU and the training the rest.


If you want to know more about the needed workflow to set up teacher scores & models: [distillation_workflow_dual-supervision.md](distillation_workflow_dual-supervision.md)

## Indexing & FAISS Vector Retrieval

If you want to know more about the evaluation options see: [dense_retrieval_evaluate.md](dense_retrieval_evaluate.md)
