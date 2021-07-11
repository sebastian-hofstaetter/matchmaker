#
# train a neural-ir model
# -------------------------------
#
# features:
#
# * uses pytorch + allenNLP
# * tries to correctly encapsulate the data source (msmarco)
# * clear configuration with yaml files
#
# usage:
# python train.py --run-name experiment1 --config-file configs/knrm.yaml

import argparse
import copy
import os
import gc
import glob
import time
import sys
sys.path.append(os.getcwd())
import itertools

# needs to be before torch import 
from allennlp.common import Params, Tqdm

#import line_profiler
#import line_profiler_py35
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
import numpy
import random

from allennlp.nn.util import move_to_device
from matchmaker.utils.utils import *
from matchmaker.utils.config import *

from matchmaker.models.all import get_model, get_word_embedder, build_model

from typing import Dict, Tuple, List
from matchmaker.utils.input_pipeline import *
from matchmaker.utils.performance_monitor import * 
from matchmaker.eval import *
from torch.utils.tensorboard import SummaryWriter

Tqdm.default_mininterval = 10000

if __name__ == "__main__":

    #
    # config
    #
    args = get_parser().parse_args()
    from_scratch = True
    if args.continue_folder:
        from_scratch = False
        run_folder = args.continue_folder
        config = get_config_single(os.path.join(run_folder, "config.yaml"), args.config_overwrites)
    else:
        config = get_config(args.config_file, args.config_overwrites)
        run_folder = prepare_experiment(args, config)

    logger = get_logger_to_file(run_folder, "main")
    logger.info("Running: %s", str(sys.argv))
    tb_writer = SummaryWriter(run_folder)
    print_hello(config,run_folder,"Teacher-Train-Score")

    #
    # random seeds
    #
    torch.manual_seed(config["random_seed"])
    numpy.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    logger.info("Torch seed: %i ",torch.initial_seed())

    # hardcode gpu usage
    cuda_device = 0 # always take the first -> set others via cuda flag in bash
    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("startup")
    
    word_embedder, padding_idx = get_word_embedder(config)
    model, encoder_type = get_model(config,word_embedder,padding_idx)
    model = build_model(model,encoder_type,word_embedder,config)
    model = model.cuda()

    #
    # warmstart model 
    #
    if "warmstart_model_path" in config:
        load_result = model.load_state_dict(torch.load(config["warmstart_model_path"]),strict=False)
        logger.info('Warmstart init model from:  %s', config["warmstart_model_path"])
        logger.info(load_result)
        print("Warmstart Result:",load_result)

    logger.info('Model %s total parameters: %s', config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)

    use_title_body_sep = config["use_title_body_sep"]
    use_cls_scoring = config["use_cls_scoring"]
    train_sparsity = config["minimize_sparsity"]
    if train_sparsity:
        config["sparsity_log_path"] = os.path.join(run_folder, config["sparsity_log_path"])
    train_qa_spans = config["train_qa_spans"]
    use_in_batch_negatives = config["in_batch_negatives"]
    chunked_passage_scores = False

    use_submodel_caching = "submodel_train_cache_path" in config
    if use_submodel_caching:
        submodel_cacher = CrossExperimentReplayCache(config["submodel_train_cache_path"],is_readonly=config["submodel_train_cache_readonly"])

    #
    # setup-multi gpu training 
    #
    is_distributed = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        is_distributed = True
    perf_monitor.set_gpu_info(torch.cuda.device_count(),torch.cuda.get_device_name())
    use_fp16 = config["use_fp16"]

    perf_monitor.stop_block("startup")
    print("Go for training ...")
    
    
    #
    # training / saving / validation loop
    # -------------------------------
    #

    scaler = torch.cuda.amp.GradScaler()

    try:
        global_i = 0

        perf_monitor.start_block("train")
        perf_start_inst = 0

        #
        # data loading 
        # -------------------------------
        #
        input_loader = allennlp_triple_training_loader(config, config, config["train_tsv"])                  

        #
        # vars we need for the training loop 
        # -------------------------------
        #
        model.eval()  # only has an effect, if we use dropout & regularization layers in the model definition...

        training_batch_size = int(config["batch_size_train"])

        concated_sequences = False
        if config["model"] == "bert_cat":
            concated_sequences = True
        
        teach_score_out = os.path.join(run_folder, "train_scores.tsv")

        #
        # train loop 
        # -------------------------------
        #
        progress = Tqdm.tqdm()
        i=0
        lastRetryI=0
        with open(teach_score_out,"w",encoding="utf8") as scores_out_file:
            for batch in input_loader:
                with torch.cuda.amp.autocast(enabled=config["use_fp16"]), \
                     torch.no_grad():

                    batch = move_to_device(batch, cuda_device)    

                    #
                    # create input 
                    #
                    pos_in = []
                    neg_in = []
                    if concated_sequences:
                        pos_in.append(batch["doc_pos_tokens"])  
                        neg_in.append(batch["doc_neg_tokens"])
                    else:
                        pos_in += [batch["query_tokens"],batch["doc_pos_tokens"]]
                        neg_in += [batch["query_tokens"],batch["doc_neg_tokens"]]

                    if use_title_body_sep:
                        pos_in.append(batch["title_pos_tokens"])
                        neg_in.append(batch["title_neg_tokens"])

                    if train_qa_spans: # add start positions for qa training (used to anchor end logits on the start ground truth)
                        pos_in.append(batch["pos_qa_start"])

                    #
                    # run model forward
                    #
                    output_pos = model.forward(*pos_in, use_fp16 = use_fp16)
                    output_neg = model.forward(*neg_in, use_fp16 = use_fp16)

                    #
                    # untangle output
                    #
                    if use_in_batch_negatives:
                        output_pos, query_vecs_pos, doc_vecs_pos = output_pos
                        output_neg, query_vecs_neg, doc_vecs_neg = output_neg

                    if train_qa_spans:
                        output_pos,answerability_pos,qa_logits_start_pos,qa_logits_end_pos = output_pos
                        output_neg,answerability_neg,qa_logits_start_neg,qa_logits_end_neg = output_neg
                    if chunked_passage_scores:
                        output_pos, chunk_scores_pos = output_pos
                        output_neg, chunk_scores_neg = output_neg
                        chunk_scores_pos = chunk_scores_pos.cpu()
                        chunk_scores_neg = chunk_scores_neg.cpu()
                        
                    ranking_score_pos = output_pos.cpu()
                    ranking_score_neg = output_neg.cpu()

                #
                # write out teacher files
                #
                for bs_i in range(ranking_score_pos.shape[0]):
                    if chunked_passage_scores:
                        scores_out_file.write(str(float(ranking_score_pos[bs_i])) + "\t" +" ".join([str(float(s)) for s in chunk_scores_pos[bs_i]]) + \
                                   "\t" + str(float(ranking_score_neg[bs_i])) + "\t" +" ".join([str(float(s)) for s in chunk_scores_neg[bs_i]]) + \
                                   "\t"+batch["query_text"][bs_i]+"\t"+batch["doc_pos_text"][bs_i]+"\t"+batch["doc_neg_text"][bs_i]+"\n")
                    else:
                        scores_out_file.write(str(float(ranking_score_pos[bs_i])) + "\t" + str(float(ranking_score_neg[bs_i])) +  \
                                   "\t"+batch["query_text"][bs_i]+"\t"+batch["doc_pos_text"][bs_i]+"\t"+batch["doc_neg_text"][bs_i]+"\n")

                progress.update()
                i+=1
                global_i+=1

            progress.close()

            perf_monitor.stop_block("train",i - perf_start_inst)
            perf_monitor.save_summary(os.path.join(run_folder,"perf-monitor.txt"))

    except:
        logger.info('-' * 89)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        print("----- Attention! - something went wrong in the train loop (see logger) ----- ")