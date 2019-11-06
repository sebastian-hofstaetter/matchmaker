# Welcome to transformer-kernel-ranking 👋



## General ideas / setup

* train.py is the main trainer -> it uses a multiprocess batch generation pipeline
* the multiprocess pipeline requires us to do some preprocessing: split the files with preprocessing/generate_file_split.sh (so that each loader process gets its own file and does not need to coordinate)

## How to train the models

1. Get the msmarco dataset & clone this repository to a pc with 1 cuda device
2. Prepare the dataset for multiprocessing:
    * Use ``./generate_file_split.sh`` 1x for training.tsv and 1x for top1000dev.tsv (the validation set)
    * You have to decide now on the number of data preparation processes you want to use for training and validation
    * You have to decide on the batch size 
    * Run ``./generate_file_split.sh <base_file> <n file chunks> <output_folder_and_prefix> <batch_size>`` for train + validation sets
    * Take the number of batches that are output at the end of the script and put them in your config .yaml
    * The number of processes for preprocessing depends on your local hardware, the preprocesses need to be faster at generating the batches then the gpu at computing the results for them (validation is much faster than training, so you need more processes)
3. Create a new config .yaml in configs/ with all your local paths + batch counts for train and validation
    * The train and validation paths should be the output folder of 2 with a star at the end (the paths will be globed to get all files)`
4. Create a new conda env and install the requirements for python 3.7 via conda: pytorch, allennlp
5. Run ``train.py`` with ``python -W ignore train.py --run-name experiment1 --config-file configs/your_file.yaml`` (-W ignore = ignores useless spacy import warnings, that come up for every subprocess (and there are many of them))