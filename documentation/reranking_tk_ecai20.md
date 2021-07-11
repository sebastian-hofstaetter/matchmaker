# Use matchmaker to train the TK model

1. Follow the getting started guide to setup the library & data format necessary for training.
2. The TK model needs embedding data (for example a glove embedding), set the settings in the dataset:
````
pre_trained_embedding: "path_to\glove42B.txt"
pre_trained_embedding_dim: 300
vocab_directory: "path_to\config\vocabs\allen_vocab_lower_glove42"
````
We provide a sample vocab for the full glove42B in the AllenNLP format in the config folder.

Example train command:
````
python matchmaker/train.py --config-file config/train/defaults.yaml config/data/<your dataset here>.yaml config/train/models/tk.yaml --run-name tk_default
````
