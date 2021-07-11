# Use matchmaker to train the BERT_CAT model

1. Follow the getting started guide to setup the library & data format necessary for training.
2. Run training:

Example train command:
````
python matchmaker/train.py --config-file config/train/defaults.yaml config/data/<your dataset here>.yaml config/train/models/bert_cat.yaml --run-name bert_cat_default
````
