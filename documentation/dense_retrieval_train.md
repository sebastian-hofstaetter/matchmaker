# Train a Standalone Dense Retriever (BERT_DOT)

BERT_DOT (a dense retrieval & ranking model using a single vector per query & doc and scoring them via dot-product) comes in two variants:

- Shared encoder: **bert_dot** (our default setting, having sep. encoders only increases memory consumption)
- Separated q,d encoders: **bert_dot_dualencoder** (usable with DPR weights)


**Important config settings**
````
#model: "bert_dot" # for shared bert model weights (q & d = the same)
model: "bert_dot_dualencoder" # for separated bert model weights (d = bert_pretrained_model, q = bert_pretrained_model_secondary)

# for dpr:
bert_pretrained_model: "facebook/dpr-ctx_encoder-multiset-base"
bert_pretrained_model_secondary: "facebook/dpr-question_encoder-multiset-base"
````

Those are not the only config settings that need to be set, in general all setting keys that are in configs/defaults.yaml should be loaded every time to ensure the code working properly and not breaking halfway through training


## Ranking training & Re-ranking Evaluation

For the dataset config, see the *config/train/data/example-minimal-dataset.yaml*.

Example train command (cd in ir-project-matchmaker):
````
CUDA_VISIBLE_DEVICES=0,1,2,3 python matchmaker/train.py --config-file config/train/defaults.yaml config/train/data/<your dataset here>.yaml config/train/models/bert_dot.yaml --run-name your_experiment_name
````

If you want to use Margin-MSE pairwise knowledge distillation, you have to set the following config files + change the train_tsv:

````
train_pairwise_distillation: True
loss: "margin-mse"
````

If you want to know more about the needed workflow see: [distillation_workflow_pairwise-supervision.md](distillation_workflow_pairwise-supervision.md)

## Indexing & FAISS Vector Retrieval

If you want to know more about the evaluation options see: [dense_retrieval_evaluate.md](dense_retrieval_evaluate.md)
