# Configuration System

The matchmaker library supports a large range of possible configurations, therefore we opted for a file-based configuration, rather than a command-line flag based. 

The ``train.py --config-file`` option allows for a number of configuration files in the yaml format, that are merged (configurations in later files override previous values). We recommend using the defaults + 1 file for collection specific config paths + 1 file for experiment specific model & training hyperparameters. 

Additionally, for bulk-experiments (or running ablation studies, etc..) the ``train.py --config-overwrites`` option allows to overwrite individual config values of the files in the form of "key1: valueA,key2: valueB". (The whitespaces are important, as the values are parsed via yaml)