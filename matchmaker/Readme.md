# Matchmaker Runners

We have the following main points of running experiments in the matchmaker library:

- **[train.py](train.py)** Main training runner for both re-ranking & retrieval training of all models supported by matchmaker.

- **[eval.py](eval.py)** Allows to evaluate a re-ranking model. 

- **[dense-retrieval.py](dense-retrieval.py)** Allows to encode, index, and evaluate a trained dense retrieval model

- **[pre-train.py](pre-train.py)** Currently a bit abandoned, but allows to pre-train model weights on an mlm or other self-supervised pre-training task
