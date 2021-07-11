# Use matchmaker to train the TKL model

1. Follow the getting started guide to setup the library & data format necessary for training.
2. The TKL model needs embedding data (for example a glove embedding), set the settings in the dataset:
````
pre_trained_embedding: "path_to\glove42B.txt"
pre_trained_embedding_dim: 300
vocab_directory: "path_to\config\vocabs\allen_vocab_lower_glove42"
````

3. Usually, you want to use TKL for long text, therefore also set the maximum doc length in the config:
````
max_doc_length: 2000
````

Example train command:
````
python matchmaker/train.py --config-file config/train/defaults.yaml config/data/<your dataset here>.yaml config/train/models/tkl.yaml --run-name tkl_default
````