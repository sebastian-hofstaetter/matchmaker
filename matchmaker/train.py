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

# needs to be before torch import 
from allennlp.common import Params, Tqdm

#import line_profiler
#import line_profiler_py35
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.multiprocessing as mp
import numpy
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import move_to_device
#from allennlp.training.trainer import Trainer

from utils import *
from models.tk import *
from matchmaker.models.tk_native import *
from models.knrm import KNRM
from models.conv_knrm import Conv_KNRM
from models.matchpyramid import MatchPyramid
from models.mv_lstm import MV_LSTM
from models.pacrr import PACRR
from models.co_pacrr import CO_PACRR
from models.duet import Duet
from models.drmm import DRMM
from models.bert_cls import *

from matchmaker.modules.neuralIR_encoder import *
from matchmaker.modules.fasttext_token_embedder import *

from dataloaders.fasttext_token_indexer import *
from dataloaders.ir_triple_loader import *
from dataloaders.ir_labeled_tuple_loader import IrLabeledTupleDatasetReader
from evaluation.msmarco_eval import *
from typing import Dict, Tuple, List
from multiprocess_input_pipeline import *
from matchmaker.performance_monitor import * 
from matchmaker.eval import *

#from apex import amp
Tqdm.default_mininterval = 1

#
# main process
# -------------------------------
#
if __name__ == "__main__":

    #
    # config
    # -------------------------------
    #
    args = get_parser().parse_args()

    from_scratch = True
    if args.continue_folder:
        from_scratch = False
        run_folder = args.continue_folder
        config = get_config_single(os.path.join(run_folder, "config.yaml"), args.config_overwrites)

    if from_scratch:
        # the config object should only be used in this file, to keep an overview over the usage
        config = get_config(args.config_file, args.config_overwrites)
        run_folder = prepare_experiment(args, config)

    logger = get_logger_to_file(run_folder, "main")

    logger.info("Running: %s", str(sys.argv))
    
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
    
    #
    # create (or load) model instance
    # -------------------------------
    #
    # * vocab (pre-built, to make the embedding matrix smaller, see generate_vocab.py)
    # * pre-trained embedding
    # * network
    # * optimizer & loss function
    #

    #
    # load candidate set for efficient cs@N validation 
    #
    validation_cont_candidate_set = parse_candidate_set(config["validation_cont"]["candidate_set_path"],config["validation_cont"]["candidate_set_from_to"][1])

    # embedding layer (use pre-trained, but make it trainable as well)
    if config["token_embedder_type"] == "embedding":
        vocab = Vocabulary.from_files(config["vocab_directory"])
        tokens_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["pre_trained_embedding"],
                                                              "embedding_dim": config["pre_trained_embedding_dim"],
                                                              "trainable": config["train_embedding"],
                                                              "padding_index":0,
                                                              "sparse":config["sparse_gradient_embedding"]}))
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
        
    elif config["token_embedder_type"] == "fasttext":
        vocab = None #FastTextVocab(config["fasttext_vocab_mapping"])
        tokens_embedder = FastTextEmbeddingBag(numpy.load(config["fasttext_weights"]),sparse=True,requires_grad=config["train_embedding"],mode=config["fasttext_merge_mode"])
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder},allow_unmatched_keys = True,embedder_to_indexer_map={"tokens":{"tokens":"tokens","offsets":"offsets"}})

    elif config["token_embedder_type"] == "elmo":
        vocab = None
        tokens_embedder = ElmoTokenEmbedder(config["elmo_options_file"],config["elmo_weights_file"])
        word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})
    elif config["token_embedder_type"] == "bert_cls":
        pass
    else:
        logger.error("token_embedder_type %s not known",config["token_embedder_type"])
        exit(1)

    encoder_type = NeuralIR_Encoder

    if config["model"] == "TK_v1": model = TK_v1.from_config(config,word_embedder.get_output_dim())
    elif config["model"] == "TK_v2": model = TK_v2.from_config(config,word_embedder.get_output_dim())

    elif config["model"] == "TK_Native_v1": model = TK_Native_v1.from_config(config,word_embedder.get_output_dim())

    #
    # baselines with text only
    #
    elif config["model"] == "knrm": model = KNRM.from_config(config,word_embedder.get_output_dim())
    elif config["model"] == "conv_knrm": model = Conv_KNRM.from_config(config,word_embedder.get_output_dim())
    elif config["model"] == "match_pyramid": model = MatchPyramid.from_config(config,word_embedder.get_output_dim())

    #
    # baseline models with idf use
    #
    elif config["model"] == "pacrr":
        model = PACRR.from_config(config,word_embedder.get_output_dim())
        encoder_type = NeuralIR_Encoder_WithIdfs
    elif config["model"] == "co_pacrr":
        model = CO_PACRR.from_config(config,word_embedder.get_output_dim())
        encoder_type = NeuralIR_Encoder_WithIdfs
    elif config["model"] == "duet":
        model = Duet.from_config(config,word_embedder.get_output_dim())
        encoder_type = NeuralIR_Encoder_WithIdfs

    #
    # bert models
    #
    elif config["model"] == "bert_cls":
        model = Bert_cls(vocab=None,bert_model = config["bert_pretrained_model"],trainable=config["bert_trainable"])
        encoder_type = None

    elif config["model"] == "drmm":
        model = DRMM(word_embedder,10).cuda(cuda_device)

    else:
        logger.error("Model %s not known",config["model"])
        exit(1)

    if encoder_type == None:
        pass
    elif encoder_type == NeuralIR_Encoder_WithIdfs:
        if config["token_embedder_type"] == "embedding":
            idf_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["idf_path"],
                                                                "embedding_dim": 1,
                                                                "trainable": config["idf_trainable"],
                                                                "padding_index":0}))
            idf_embedder = BasicTextFieldEmbedder({"tokens":idf_embedder})
        elif config["token_embedder_type"] == "fasttext":
            idf_embedder = FastTextEmbeddingBag(numpy.load(config["idf_path_fasttext"]),
                                                requires_grad=config["idf_trainable"],
                                                mode="mean")
            idf_embedder = BasicTextFieldEmbedder({"tokens": idf_embedder}, 
                                                  allow_unmatched_keys = True, 
                                                  embedder_to_indexer_map={"tokens":{"tokens":"tokens","offsets":"offsets"}})

        model = encoder_type(word_embedder, idf_embedder, model)    
    else:
        model = encoder_type(word_embedder, model)

    model = model.cuda()

    #
    # warmstart model 
    #
    if "warmstart_model_path" in config:
        logger.info('Warmstart init model from:  %s', config["warmstart_model_path"])
        model.load_state_dict(torch.load(config["warmstart_model_path"]))

    logger.info('Model %s total parameters: %s', config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)


    params_group0 = []
    params_group1 = []
    we_params = None

    for p_name,par in model.named_parameters():
        spl=p_name.split(".")
        group1_check = p_name
        if len(spl) >= 2:
            group1_check = spl[1]

        if not config["train_embedding"] and p_name.startswith("word_embedding"):
            pass
        elif config["train_embedding"] and p_name.startswith("word_embedding") and not config["use_fp16"]:
            we_params = par
            logger.info("we_params: %s",p_name)
        elif group1_check in config["param_group1_names"]:
            params_group1.append(par)
            logger.info("params_group1: %s",p_name)
        else:
            params_group0.append(par)

    all_params =[
        {"params":params_group0,"lr":config["param_group0_learning_rate"],"weight_decay":config["param_group0_weight_decay"]},
        {"params":params_group1,"lr":config["param_group1_learning_rate"],"weight_decay":config["param_group1_weight_decay"]}
    ]

    embedding_optimizer=None
    use_embedding_optimizer = False
    if type(we_params) != None and config["train_embedding"] and not config["use_fp16"]:
        use_embedding_optimizer = True
        if config["embedding_optimizer"] == "adam":
            embedding_optimizer = Adam([we_params], lr=config["embedding_optimizer_learning_rate"])

        elif config["embedding_optimizer"] == "sparse_adam":
            embedding_optimizer = SparseAdam([we_params], lr=config["embedding_optimizer_learning_rate"])

        elif config["embedding_optimizer"] == "sgd":
            embedding_optimizer = SGD([we_params], lr=config["embedding_optimizer_learning_rate"],momentum=config["embedding_optimizer_momentum"])

    if config["optimizer"] == "adam":
        optimizer = Adam(all_params)

    elif config["optimizer"] == "sgd":
        optimizer = SGD(all_params, momentum=0.5)

    #
    # fp 16 support via nvidia apex
    #
    #model, optimizer = amp.initialize(model,optimizer, opt_level="O2",enabled=config["use_fp16"])


    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                     patience=config["learning_rate_scheduler_patience"],
                                     factor=config["learning_rate_scheduler_factor"],
                                     verbose=True)
    early_stopper = EarlyStopping(patience=config["early_stopping_patience"], mode="max")

    criterion = torch.nn.MarginRankingLoss(margin=1, reduction='mean').cuda(cuda_device)

    loss_file_path = os.path.join(run_folder, "training-loss.csv")
    # write csv header once
    with open(loss_file_path, "w") as loss_file:
        loss_file.write("sep=,\nEpoch,After_Batch,Loss\n")
    
    best_metric_info_file = os.path.join(run_folder, "best-info.csv")
    best_model_store_path = os.path.join(run_folder, "best-model.pytorch-state-dict")

    # keep track of the best metric
    if from_scratch:
        best_metric_info = {}
        best_metric_info["metrics"]={}
        best_metric_info["metrics"][config["validation_metric"]] = 0
    else:
        best_metric_info = read_best_info(best_metric_info_file)

    perf_monitor.stop_block("startup")
    model.get_param_stats() # just test that it works (very much needed for new models ^^)
    
    #
    # training / saving / validation loop
    # -------------------------------
    #
    #@profile
    #def work():
    try:
        if from_scratch:
            for epoch in range(0, int(config["epochs"])):
                if early_stopper.stop:
                    break
                perf_monitor.start_block("train")
                perf_start_inst = 0
                #
                # data loading 
                # -------------------------------
                #
                training_queue, training_processes, train_exit = get_multiprocess_batch_queue("train-batches-" + str(epoch),
                                                                                              multiprocess_training_loader,
                                                                                              glob.glob(config.get("train_tsv")),
                                                                                              config,
                                                                                              logger)
                #time.sleep(len(training_processes))  # fill the queue
                logger.info("[Epoch "+str(epoch)+"] --- Start training with queue.size:" + str(training_queue.qsize()))

                #
                # vars we need for the training loop 
                # -------------------------------
                #
                model.train()  # only has an effect, if we use dropout & regularization layers in the model definition...
                loss_sum = torch.zeros(1).cuda(cuda_device)
                training_batch_size = int(config["batch_size_train"])
                # label is always set to 1 - indicating first input is pos (see criterion:MarginRankingLoss) + cache on gpu
                label = torch.ones(training_batch_size).cuda(cuda_device)

                # helper vars for quick checking if we should validate during the epoch
                validate_every_n_batches = config["validate_every_n_batches"]
                do_validate_every_n_batches = validate_every_n_batches > -1

                #s_pos = torch.cuda.Stream()
                #s_neg = torch.cuda.Stream()
                concated_sequences = False
                if config["model"] == "bert_cls":
                    concated_sequences = True

                gradient_accumulation = False
                if config["gradient_accumulation_steps"] > 0:
                    gradient_accumulation = True
                    gradient_accumulation_steps = config["gradient_accumulation_steps"]

                #
                # train loop 
                # -------------------------------
                #
                #for i in Tqdm.tqdm(range(0, config["training_batch_count"]), disable=config["tqdm_disabled"]):
                progress = Tqdm.tqdm()
                i=0
                lastRetryI=0
                while True:
                    try:
                        batch = training_queue.get()
                        if batch == None: # this means we got it all
                            break

                        current_batch_size = batch["doc_pos_tokens"]["tokens"].shape[0]

                        batch = move_to_device(batch, cuda_device)             
                        if concated_sequences:
                            output_pos = model.forward(batch["doc_pos_tokens"])
                            output_neg = model.forward(batch["doc_neg_tokens"])
                        else:
                            #with torch.cuda.stream(s_pos):
                            output_pos = model.forward(batch["query_tokens"], batch["doc_pos_tokens"])
                            #with torch.cuda.stream(s_neg):
                            output_neg = model.forward(batch["query_tokens"], batch["doc_neg_tokens"])
                        #torch.cuda.synchronize()

                        # the last batches might (will) be smaller, so we need to check the batch size :(
                        # but it should only affect the last n batches (n = # data loader processes) so we don't need a performant solution
                        if current_batch_size != training_batch_size:
                            label = torch.ones(current_batch_size).cuda(cuda_device)

                        loss = criterion(output_pos, output_neg, label)
                        #loss = torch.mean(- torch.log(output_pos) - torch.log(1 - output_neg))
                        #with amp.scale_loss(loss, optimizer) as scaled_loss:
                        #    scaled_loss.backward()
                        loss.backward()

                        if gradient_accumulation:
                            if (1+i)%gradient_accumulation_steps == 0:
                                optimizer.step()
                                if use_embedding_optimizer:
                                    embedding_optimizer.step()
                                optimizer.zero_grad()
                                if use_embedding_optimizer:
                                    embedding_optimizer.zero_grad()
                                #torch.cuda.empty_cache() # bad perf, allow bert to run on 1080!
                        else:
                            optimizer.step()
                            if use_embedding_optimizer:
                                embedding_optimizer.step()
                            optimizer.zero_grad()
                            if use_embedding_optimizer:
                                embedding_optimizer.zero_grad()
                        # set the label back to a cached version (for next iterations)
                        if current_batch_size != training_batch_size:
                            label = torch.ones(training_batch_size).cuda(cuda_device)

                        loss_sum = loss_sum.data + loss.detach().data
                        #loss = loss.detach()
                        #del loss, output_neg, output_pos, batch
                        #torch.cuda.synchronize() # only needed for profiling to get the remainder of the cuda work done and not put into another line

                        if i > 0 and i % 100 == 0:
                            # append loss to loss file
                            with open(loss_file_path, "a") as loss_file:
                                loss_file.write(str(epoch) + "," +str(i) + "," + str(loss_sum.item()/100) +"\n")

                            # reset sum (on gpu)
                            loss_sum = torch.zeros(1).cuda(cuda_device)

                            # make sure that the perf of the queue is sustained
                            if training_queue.qsize() < 10:
                                logger.warning("training_queue.qsize() < 10")

                    except RuntimeError as r:
                        if r.args[0].startswith("CUDA out of memory"): # lol yeah that is python for you 
                            if i - lastRetryI < 2:
                                raise r

                            del loss,output_neg,output_pos,batch
                            gc.collect()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            time.sleep(5)
                            lastRetryI=i
                            logger.warning("["+str(i)+"] Caught CUDA OOM: " + r.args[0]+", now cached:"+str(torch.cuda.memory_cached()))
                        else:
                            raise r
                            
                    #
                    # vars we need for the training loop 
                    # -------------------------------
                    #
                    if do_validate_every_n_batches:
                        if i > 0 and i % validate_every_n_batches == 0:

                            perf_monitor.stop_block("train",(i - perf_start_inst) * training_batch_size)
                            perf_start_inst = i
                            perf_monitor.start_block("cont_val")

                            best_metric, _, validated_count = validate_model("cont",model, config,config["validation_cont"], run_folder, logger, cuda_device, epoch, i,
                                                                             best_metric_info,validation_cont_candidate_set,use_cache=config["validation_cont_use_cache"],output_secondary_output=False)
                            if best_metric["metrics"][config["validation_metric"]] > best_metric_info["metrics"][config["validation_metric"]]:
                                best_metric_info = best_metric
                                save_best_info(best_metric_info_file,best_metric_info)
                                torch.save(model.state_dict(), best_model_store_path)
                                logger.info(str(i)+"Saved new best weights with: "+config["validation_metric"]+": " + str(best_metric["metrics"][config["validation_metric"]]))

                            logger.info(str(i)+"-"+model.get_param_stats())

                            #if config["model"] == "knrm" or config["model"] == "knrm_ln" or config["model"] == "conv_knrm" or config["model"] == "conv_knrm_same_gram":
                            #    #logger.info("KNRM-dense layer: b: %s , weight: %s",str(model.dense.bias.data), str(model.dense.weight.data))
                            #    logger.info("KNRM-dense layer: weight: %s",str(model.dense.weight.data))
                            #if config["model"] in ["knrm_ln","conv_knrm_ln"]:
                            #    logger.info("KNRM_LN-length_norm_factor: weight: %s",str(model.length_norm_factor.data))

                            model.train()
                            lr_scheduler.step(best_metric["metrics"][config["validation_metric"]])
                            if early_stopper.step(best_metric["metrics"][config["validation_metric"]]):
                                logger.info("early stopping epoch %i batch count %i",epoch,i)
                                break

                            perf_monitor.stop_block("cont_val",validated_count)
                            perf_monitor.start_block("train")
                            perf_monitor.print_summary()

                    progress.update()
                    i+=1
                progress.close()
                # make sure we didn't make a mistake in the configuration / data preparation
                if training_queue.qsize() != 0:
                    logger.error("training_queue.qsize() is not empty after epoch "+str(epoch))

                train_exit.set()  # allow sub-processes to exit
                perf_monitor.stop_block("train",i - perf_start_inst)

                #
                # validation (after epoch)
                #
                #best_metric, _, validated_count = validate_model("cont",model, config, run_folder, logger, cuda_device, epoch,-1,best_metric_info,validation_cont_candidate_set,use_cache=True)
                #if best_metric["metrics"]["MRR"] > best_metric_info["metrics"]["MRR"]:
                #    best_metric_info = best_metric
                #    save_best_info(best_metric_info_file,best_metric_info)
                #    torch.save(model.state_dict(), best_model_store_path)

                #if config["model"] == "knrm" or config["model"] == "conv_knrm" or config["model"] == "conv_knrm_same_gram":
                    #logger.info("KNRM-dense layer: b: %s , weight: %s",str(model.dense.bias.data), str(model.dense.weight.data))
                #    logger.info("KNRM-dense layer: weight: %s",str(model.dense.weight.data))
                #logger.info(model.get_param_stats())

                #lr_scheduler.step(best_metric["metrics"]["MRR"])
                #if early_stopper.step(best_metric["metrics"]["MRR"]):
                #    logger.info("early stopping epoch %i",epoch)
                #    break

        #
        # evaluate the 2nd validation (larger) & test set with the best model
        #
        print("Mem allocated:",torch.cuda.memory_allocated())

        perf_monitor.log_value("train_gpu_mem",str(torch.cuda.memory_allocated()/float(1e9)) + " GB")
        perf_monitor.log_value("train_gpu_mem_max",str(torch.cuda.max_memory_allocated()/float(1e9)) + " GB")

        model_cpu = model.cpu() # we need this strange back and forth copy for models > 1/2 gpu memory, because load_state copies the state dict temporarily
        del model, optimizer, all_params, we_params, lr_scheduler
        if from_scratch:
            del loss
        if config["train_embedding"]:
            del embedding_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(10) # just in case the gpu has not cleaned up the memory
        torch.cuda.reset_max_memory_allocated()
        model_cpu.load_state_dict(torch.load(best_model_store_path,map_location="cpu"))
        model = model_cpu.cuda(cuda_device)
        print("Model reloaded ! mem:",torch.cuda.memory_allocated())

        best_validation_end_metrics={}
        if "validation_end" in config:
            for validation_end_name,validation_end_config in config["validation_end"].items():
                validation_end_candidate_set = parse_candidate_set(validation_end_config["candidate_set_path"],validation_end_config["candidate_set_from_to"][1])
                best_metric, _, validated_count = validate_model(validation_end_name,model, config,validation_end_config,
                                                                 run_folder, logger, cuda_device, 
                                                                 candidate_set=validation_end_candidate_set,
                                                                 output_secondary_output=validation_end_config["save_secondary_output"])
                save_best_info(os.path.join(run_folder, "val-"+validation_end_name+"-info.csv"),best_metric)
                best_validation_end_metrics[validation_end_name] = best_metric

        if "test" in config:
            for test_name,test_config in config["test"].items():
                cs_at_n_test = best_validation_end_metrics[test_name]["cs@n"]
                test_candidate_set = parse_candidate_set(test_config["candidate_set_path"],test_config["candidate_set_max"])
                test_result = test_model(model, config,test_config, run_folder, logger, cuda_device,"test_"+test_name,
                                         test_candidate_set, cs_at_n_test,output_secondary_output=test_config["save_secondary_output"])

        if "leaderboard" in config:
            for test_name,test_config in config["leaderboard"].items():
                test_model(model, config, test_config, run_folder, logger, cuda_device, "leaderboard"+test_name,output_secondary_output=test_config["save_secondary_output"])

        perf_monitor.log_value("eval_gpu_mem",str(torch.cuda.memory_allocated()/float(1e9)) + " GB")
        perf_monitor.log_value("eval_gpu_mem_max",str(torch.cuda.max_memory_allocated()/float(1e9)) + " GB")

        perf_monitor.save_summary(os.path.join(run_folder,"perf-monitor.txt"))

    except:
        logger.info('-' * 89)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        print("----- Attention! - something went wrong in the train loop (see logger) ----- ")

        for proc in training_processes:
            if proc.is_alive():
                proc.terminate()
        exit(1)

    # make sure to exit processes (for early stopping)
    for proc in training_processes:
        if proc.is_alive():
            proc.terminate()
    exit(0)
    
    #prof = line_profiler.LineProfiler()
#
    #prof.add_function(work)
    #prof.runcall(work)
#
    #prof.dump_stats("line_prof_inline.prof")
#
    #prof.print_stats()