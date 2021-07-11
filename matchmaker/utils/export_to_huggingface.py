#
# export a trained model to huggingface format
# -------------------------------


import argparse
import os
import sys
sys.path.append(os.getcwd())


import torch

from matchmaker.utils import *

from matchmaker.models.all import get_model, get_word_embedder, build_model

from typing import Dict, Tuple, List
from matchmaker.utils.core_metrics import *
from matchmaker.utils.config import *
from transformers import AutoTokenizer

if __name__ == "__main__":


    #
    # config
    # -------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', action='store', dest='output_folder',
                        help='Folder to output the export data ', required=True)
                        
    parser.add_argument('--checkpoint-folder', action='store', dest='continue_folder',
                        help='Experiment folder to load model from', required=True)


    args = parser.parse_args()

    from_scratch = False
    run_folder = args.continue_folder
    config = get_config_single(os.path.join(run_folder, "config.yaml"))

    model_config = config

   
    #
    # create and load model instance
    # -------------------------------
    
    word_embedder, padding_idx = get_word_embedder(model_config)
    model, encoder_type = get_model(model_config,word_embedder,padding_idx)
    model = build_model(model,encoder_type,word_embedder,model_config)

    model_path = os.path.join(run_folder, "best-model.pytorch-state-dict")
    load_result = model.load_state_dict(torch.load(model_path),strict=False)
    print("Warmstart Result:",load_result)

    if config["model"] == 'bert_dot':
        tokenizer = AutoTokenizer.from_pretrained(config["bert_pretrained_model"])
        tokenizer.save_pretrained(args.output_folder)
        model.bert_model.save_pretrained(args.output_folder)

    elif config["model"] == 'bert_cat' or config["model"] == 'bert_cls':
        tokenizer = AutoTokenizer.from_pretrained(config["bert_pretrained_model"])
        tokenizer.save_pretrained(args.output_folder)
        model.save_pretrained(args.output_folder)

    elif config["model"] == 'ColBERT':
        tokenizer = AutoTokenizer.from_pretrained(config["bert_pretrained_model"])
        tokenizer.save_pretrained(args.output_folder)
        model.save_pretrained(args.output_folder)

    elif config["model"] ==  "Bert_patch" or config["model"] ==  "IDCM":
        tokenizer = AutoTokenizer.from_pretrained(config["bert_pretrained_model"])
        tokenizer.save_pretrained(args.output_folder)
        model.save_pretrained(args.output_folder)

    elif config["model"] == "PreTTR" or config["model"] ==  "Bert_Split":
        tokenizer = AutoTokenizer.from_pretrained(config["bert_pretrained_model"])
        tokenizer.save_pretrained(args.output_folder)
        model.save_pretrained(args.output_folder)


    else:
        print("Model export not supported",config["model"])
        exit(1)
