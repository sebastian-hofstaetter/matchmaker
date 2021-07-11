# Use matchmaker to train the ColBERT model

1. Follow the getting started guide to setup the library & data format necessary for training.
2. Run training:

Example train command:
````
python matchmaker/train.py --config-file config/train/defaults.yaml config/data/<your dataset here>.yaml config/train/models/colbert.yaml --run-name colbert_default
````
