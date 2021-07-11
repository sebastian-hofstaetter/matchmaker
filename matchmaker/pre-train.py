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
from torch import autograd
import torch.multiprocessing as mp
import numpy
import random

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import ElmoTokenEmbedder, PretrainedTransformerEmbedder
from matchmaker.modules.bert_embedding_token_embedder import BertEmbeddingTokenEmbedder

from allennlp.nn.util import move_to_device
#from allennlp.training.trainer import Trainer

from matchmaker.utils import *

from matchmaker.modules.pre_train_heads import *
from typing import Dict, Tuple, List
from matchmaker.utils.multiprocess_input_pipeline import *
from matchmaker.utils.performance_monitor import * 
from matchmaker.eval import *
import transformers
from matchmaker.modules.parallel import * 
#from apex import amp
Tqdm.default_mininterval = 1

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matchmaker.models.all import get_model, get_word_embedder, build_model

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
    
    print_hello(config,run_folder,"Pre-Train")

    logger = get_logger_to_file(run_folder, "main")

    logger.info("Running: %s", str(sys.argv))
    tb_writer = SummaryWriter(run_folder)
    #tb_writer.add_hparams(config)
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


    model = PreTrain_MLM_Head(model,representation_size=384, vocab_size=model.bert_model.config.vocab_size)


    model = model.cuda()

    #print("emb: mean-len",torch.mean(torch.norm(word_embedder.token_embedder_tokens.weight.data, p=2, dim=-1)))
    #print("pos: mean-len",torch.mean(torch.norm(model.neural_ir_model.neural_ir_model.positional_features_d.data, p=2, dim=-1)))
    #emb_norm = torch.norm(model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data, p=2, dim=-1).mean()
    #model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data.normal_(mean=0.0, std=0.2)
    #model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data /= (torch.norm(model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data, p=2, dim=-1).unsqueeze(-1)+0.00001)
    #model.neural_ir_model.neural_ir_model.positional_features_d.data /= (torch.norm(model.neural_ir_model.neural_ir_model.positional_features_d.data, p=2, dim=-1).unsqueeze(-1)+0.00001)
    #model.neural_ir_model.neural_ir_model.positional_features_q.data /= (torch.norm(model.neural_ir_model.neural_ir_model.positional_features_q.data, p=2, dim=-1).unsqueeze(-1)+0.00001)
    #model.neural_ir_model.neural_ir_model.positional_features_t.data /= (torch.norm(model.neural_ir_model.neural_ir_model.positional_features_t.data, p=2, dim=-1).unsqueeze(-1)+0.00001)

    #emb_norm = torch.norm(model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data, p=2, dim=-1).mean()
    #model.neural_ir_model.neural_ir_model.positional_features_d.data /= ((torch.norm(model.neural_ir_model.neural_ir_model.positional_features_d.data, p=2, dim=-1).unsqueeze(-1)+0.00001) / emb_norm)
    #model.neural_ir_model.neural_ir_model.positional_features_q.data /= ((torch.norm(model.neural_ir_model.neural_ir_model.positional_features_q.data, p=2, dim=-1).unsqueeze(-1)+0.00001) / emb_norm)
    #model.neural_ir_model.neural_ir_model.positional_features_t.data /= ((torch.norm(model.neural_ir_model.neural_ir_model.positional_features_t.data, p=2, dim=-1).unsqueeze(-1)+0.00001) / emb_norm)
#
#
    #tb_writer.add_scalar("Model/Embedding Mean Norm",torch.mean(torch.norm(model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data, p=2, dim=-1)))
    #tb_writer.add_scalar("Model/Pos.Enc.D Mean Norm",torch.mean(torch.norm(model.neural_ir_model.neural_ir_model.positional_features_d.data, p=2, dim=-1)))
    #tb_writer.add_scalar("Model/Pos.Enc.Q Mean Norm",torch.mean(torch.norm(model.neural_ir_model.neural_ir_model.positional_features_q.data, p=2, dim=-1)))

    #
    # warmstart model 
    #
    #if "warmstart_model_path" in config:
    #    logger.info('Warmstart init model from:  %s', config["warmstart_model_path"])
    #    model.load_state_dict(torch.load(config["warmstart_model_path"]))

    logger.info('Model %s total parameters: %s', config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)



    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["pretrain_weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["pretrain_lr"])


    #
    # fp 16 support via nvidia apex
    #
    #model, optimizer = amp.initialize(model,optimizer, opt_level="O2",enabled=config["use_fp16"])


    #lr_scheduler = ReduceLROnPlateau(optimizer, mode="max",
    #                                 patience=config["learning_rate_scheduler_patience"],
    #                                 factor=config["learning_rate_scheduler_factor"],
    #                                 verbose=True)
    
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,8,30_00)


    logger.info("0: "+model.get_param_stats())
    is_distributed = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        is_distributed = True

    loss_file_path = os.path.join(run_folder, "training-loss.csv")
    # write csv header once
    with open(loss_file_path, "w") as loss_file:
        loss_file.write("sep=,\nEpoch,After_Batch,Loss\n")
    
    best_model_store_path = os.path.join(run_folder, "best-model.pytorch-state-dict")

    perf_monitor.stop_block("startup")
    #model.get_param_stats() # just test that it works (very much needed for new models ^^)
    
    #
    # training / saving / validation loop
    # -------------------------------
    #
    #@profile
    #def work():
    from collections import defaultdict
    token_masked_counter = defaultdict(int)
    sum_tokens=0
    scaler = torch.cuda.amp.GradScaler()

    try:
        loss_sum = torch.zeros(1).cuda(cuda_device)
        loss_title_sum = torch.zeros(1).cuda(cuda_device)
        perp_sum = torch.zeros(1).cuda(cuda_device)
        acc_sum = torch.zeros(1).cuda(cuda_device)
        acc2_sum = torch.zeros(1).cuda(cuda_device)
        acc2_ratio_sum = torch.zeros(1).cuda(cuda_device)
        in_doc_loss_sum = torch.zeros(1).cuda(cuda_device)
        pod_pos_mean_sum = torch.zeros(1).cuda(cuda_device)
        pod_neg_mean_sum = torch.zeros(1).cuda(cuda_device)
        title_topic_loss_sum= torch.zeros(1).cuda(cuda_device)
        title_pos_mean_sum= torch.zeros(1).cuda(cuda_device)
        title_neg_mean_sum = torch.zeros(1).cuda(cuda_device)
        inter_pod_diff_sum = torch.zeros(1).cuda(cuda_device)


        grad_norm_sum = torch.zeros(1).cuda(cuda_device)
        grad_max_sum = torch.zeros(1).cuda(cuda_device)
        grad_counter = 0

        global_i = 0
        global_update_i = 0
        gradient_update_loop = 0
        for epoch in range(0, int(config["epochs"])):

            perf_monitor.start_block("train")
            perf_start_inst = 0
            #
            # data loading 
            # -------------------------------
            #
            fs = []
            for f in config.get("pre_train_tsv").split(";"):
                fs += glob.glob(f) 
            print(fs)
            training_queue, training_processes, train_exit = get_multiprocess_batch_queue(epoch,
                                                                                          multiprocess_mlm_sequence_loader,
                                                                                          fs,
                                                                                          config,
                                                                                          logger)
            #time.sleep(len(training_processes))  # fill the queue
            logger.info("[Epoch "+str(epoch)+"] --- Start training with queue.size:" + str(training_queue.qsize()))

            #
            # vars we need for the training loop 
            # -------------------------------
            #
            model.train()  # only has an effect, if we use dropout & regularization layers in the model definition...
            training_batch_size = int(config["batch_size_pretrain"])
            # label is always set to 1 - indicating first input is pos (see criterion:MarginRankingLoss) + cache on gpu
            #label = torch.ones(training_batch_size).cuda(cuda_device)

            # helper vars for quick checking if we should validate during the epoch
            save_model_every = config["save_model_every"]


            gradient_accumulation = False
            if config["pretrain_gradient_acc_steps"] > 0:
                gradient_accumulation = True
                gradient_accumulation_steps = config["pretrain_gradient_acc_steps"]
            use_title_body_sep = False
            if config["use_title_body_sep"]:
                use_title_body_sep = True
            lr_scheduler.step()

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

                    batch = move_to_device(batch, cuda_device)             

                    #sum_tokens+=int(batch["seq_tokens_original"]["tokens"]["mask"].sum())

                    batch["seq_tokens_original"]["tokens"]["token_ids"][batch["seq_masked"] == 0] = -100
                    #batch["title_tokens_original"]["tokens"]["token_ids"][batch["title_tokens_mask"] == 0] = -100

                    #for token_id in batch["seq_tokens_original"]["tokens"]["tokens"][batch["seq_masked"] == 1].view(-1):
                    #    token_masked_counter[int(token_id)]+=1
                    if use_title_body_sep:
                        loss,perplexity,accuracy,accuracy2,loss_title,in_doc_loss,pod_pos_mean,pod_neg_mean,title_topic_loss,title_pos_mean,title_neg_mean,inter_pod_diff = model.forward(batch["seq_tokens"],batch["seq_tokens_original"]["tokens"]["tokens"],batch["seq_tf_info"],config["use_fp16"],batch["title_tokens"],batch["title_tokens_original"]["tokens"]["tokens"])
                    else:
                        loss,perplexity,accuracy,accuracy2 = model.forward(batch["seq_tokens"],batch["seq_tokens_original"]["tokens"]["token_ids"],batch["seq_tf_info"],config["use_fp16"])

                    #batch["seq_tokens_original"]["tokens"][batch["seq_tokens_original"]["tokens"] == 0] = -100

                    #loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
                    #loss = loss_fct(prediction_scores.view(-1, prediction_scores.shape[-1]), batch["seq_tokens_original"]["tokens"]["tokens"].view(-1))
                    perplexity = perplexity.mean()
                    accuracy = accuracy.mean()
                    accuracy2 = accuracy2.mean()
                    #acc2_ratio = acc2_ratio.mean()

                    loss = loss.mean()
                    #loss_title = loss_title.mean()
                    #if in_doc_loss is not None:
                    #    in_doc_loss = in_doc_loss.mean()
                    #    pod_pos_mean = pod_pos_mean.mean()
                    #    pod_neg_mean = pod_neg_mean.mean()
                    #    title_topic_loss = title_topic_loss.mean()
                    #    title_pos_mean = title_pos_mean.mean()
                    #    title_neg_mean = title_neg_mean.mean()
                    #    inter_pod_diff = inter_pod_diff.mean()
                    #    #title_topic_cosine = title_topic_cosine.mean()
                    #    #title_topic_cosine_loss = 1 - title_topic_cosine
                    
                    if gradient_accumulation:
                        #loss_title = loss_title / gradient_accumulation_steps
                        loss = loss / gradient_accumulation_steps
                        perplexity = perplexity / gradient_accumulation_steps
                        accuracy = accuracy / gradient_accumulation_steps
                        accuracy2 = accuracy2 / gradient_accumulation_steps
                        
                        #if in_doc_loss is not None:
                        #    in_doc_loss = in_doc_loss / gradient_accumulation_steps
                        #    pod_pos_mean = pod_pos_mean / gradient_accumulation_steps
                        #    pod_neg_mean = pod_neg_mean / gradient_accumulation_steps
#
                        #    title_topic_loss = title_topic_loss / gradient_accumulation_steps
                        #    title_pos_mean = title_pos_mean / gradient_accumulation_steps
                        #    title_neg_mean = title_neg_mean / gradient_accumulation_steps
                        #    inter_pod_diff = inter_pod_diff / gradient_accumulation_steps

                    #scaler.scale(loss + loss_title + (in_doc_loss + title_topic_loss)*0.5).backward() # + (in_doc_loss + title_topic_loss) * 0.5 # + (in_doc_loss) * 4 + title_topic_cosine_loss * 0.1).backward()
                    scaler.scale(loss).backward()

                    if gradient_accumulation:
                        gradient_update_loop+=1
                        if gradient_update_loop == gradient_accumulation_steps:
                            gradient_update_loop=0

                            scaler.unscale_(optimizer)
                            norm_temp = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            if not torch.isnan(norm_temp):
                                grad_counter+=1
                                grad_norm_sum = grad_norm_sum.data + norm_temp
                                grad_max_sum = grad_max_sum.data + max(p.grad.detach().abs().max() for p in model.parameters() if p.grad is not None)
                            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            global_update_i+=1
                    else:
                        scaler.unscale_(optimizer)
                        norm_temp = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        if not torch.isnan(norm_temp):
                            grad_counter+=1
                            grad_norm_sum = grad_norm_sum.data + norm_temp
                            grad_max_sum = grad_max_sum.data + max(p.grad.detach().abs().max() for p in model.parameters() if p.grad is not None)
                        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        global_update_i+=1

                    #if in_doc_loss is not None:
                    #    in_doc_loss_sum = in_doc_loss_sum.data + in_doc_loss.detach().data
                    #    pod_pos_mean_sum = pod_pos_mean_sum.data + pod_pos_mean.detach().data
                    #    pod_neg_mean_sum = pod_neg_mean_sum.data + pod_neg_mean.detach().data
                    #    
                    #    title_topic_loss_sum = title_topic_loss_sum.data +  title_topic_loss.detach().data
                    #    title_pos_mean_sum  = title_pos_mean_sum.data + title_pos_mean.detach().data
                    #    title_neg_mean_sum  = title_neg_mean_sum.data + title_neg_mean.detach().data
                    #    inter_pod_diff_sum  = inter_pod_diff_sum.data + inter_pod_diff.detach().data

                    loss_sum = loss_sum.data + loss.detach().data
                    #loss_title_sum = loss_title_sum.data + loss_title.detach().data
                    perp_sum = perp_sum.data + perplexity.detach().data
                    acc_sum = acc_sum.data + accuracy.detach().data
                    acc2_sum = acc2_sum.data + accuracy2.detach().data

                    #acc2_ratio_sum = acc2_ratio_sum.data + acc2_ratio.detach().data
                    #loss = loss.detach()
                    #del loss, output_neg, output_pos, batch
                    #torch.cuda.synchronize() # only needed for profiling to get the remainder of the cuda work done and not put into another line



                    if i > 0 and gradient_update_loop == 0 and global_update_i % 1000 == 0:
                        # append loss to loss file
                        with open(loss_file_path, "a") as loss_file:
                            loss_file.write(str(epoch) + "," +str(global_update_i) + "," + str(loss_sum.item()/1000) +"\n")
                        
                        #if in_doc_loss is not None:
                        #    tb_writer.add_scalar('Loss/In-Doc (POD)', in_doc_loss_sum.item()/1000, global_update_i)
                        #    tb_writer.add_scalar('Train/POD pos mean', pod_pos_mean_sum.item()/1000, global_update_i)
                        #    tb_writer.add_scalar('Train/POD neg mean', pod_neg_mean_sum.item()/1000, global_update_i)
                        #    
                        #    tb_writer.add_scalar('Loss/Title-Pseudo Query (TPQ)', title_topic_loss_sum.item()/1000, global_update_i)
                        #    tb_writer.add_scalar('Train/TPQ pos mean', title_pos_mean_sum.item()/1000, global_update_i)
                        #    tb_writer.add_scalar('Train/TPQ neg mean', title_neg_mean_sum.item()/1000, global_update_i)
                        #    
                        #    tb_writer.add_scalar('Train/Inter POD diff', inter_pod_diff_sum.item()/1000, global_update_i)

                        tb_writer.add_scalar('Loss/train', loss_sum.item()/1000, global_update_i)
                        #tb_writer.add_scalar('Loss/Title-MLM', loss_title_sum.item()/1000, global_update_i)
                        tb_writer.add_scalar('Train/Perplexity', perp_sum.item()/1000, global_update_i)
                        tb_writer.add_scalar('Train/MLM accuracy', acc_sum.item()/1000, global_update_i)
                        tb_writer.add_scalar('Train/MLM accuracy (lower half of median TF)', acc2_sum.item()/1000, global_update_i)
                        tb_writer.add_scalar('Model/All Gradient Norm', grad_norm_sum.item()/grad_counter, global_update_i)
                        tb_writer.add_scalar('Model/Max Gradient Avg.', grad_max_sum.item()/grad_counter, global_update_i)
                        tb_writer.add_scalar('Model/Scaler.scale', scaler.get_scale(), global_update_i)
                        
                        #tb_writer.add_scalar('Train/MLM ratio (lower half of median TF)', acc2_ratio_sum.item()/1000, global_i)

                        #if is_distributed:
                        #    tb_writer.add_scalar("Model/Embedding Mean Norm",torch.mean(torch.norm(model.module.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data, p=2, dim=-1)), global_update_i)
                        #    tb_writer.add_scalar("Model/Pos.Enc.D Mean Norm",torch.mean(torch.norm(model.module.neural_ir_model.neural_ir_model.positional_features_d.data, p=2, dim=-1)), global_update_i)
                        #    #tb_writer.add_scalar("Model/Pos.Enc.Q Mean Norm",torch.mean(torch.norm(model.module.neural_ir_model.neural_ir_model.positional_features_q.data, p=2, dim=-1)), global_update_i)
                        #else:
                        #    tb_writer.add_scalar("Model/Embedding Mean Norm",torch.mean(torch.norm(model.neural_ir_model.word_embeddings.token_embedder_tokens.weight.data, p=2, dim=-1)), global_update_i)
                        #    tb_writer.add_scalar("Model/Pos.Enc.D Mean Norm",torch.mean(torch.norm(model.neural_ir_model.neural_ir_model.positional_features_d.data, p=2, dim=-1)), global_update_i)
                        #    #tb_writer.add_scalar("Model/Pos.Enc.Q Mean Norm",torch.mean(torch.norm(model.neural_ir_model.neural_ir_model.positional_features_q.data, p=2, dim=-1)), global_update_i)
#

                        # reset sum (on gpu)
                        #in_doc_loss_sum = torch.zeros(1,device=loss_sum.device)
                        loss_sum = torch.zeros(1,device=loss_sum.device)
                        #loss_title_sum *= 0
                        perp_sum = torch.zeros(1,device=loss_sum.device)
                        acc_sum = torch.zeros(1,device=loss_sum.device)
                        acc2_sum = torch.zeros(1,device=loss_sum.device)
                        #pod_pos_mean_sum = torch.zeros(1,device=loss_sum.device)
                        #pod_neg_mean_sum = torch.zeros(1,device=loss_sum.device)

                        #title_topic_loss_sum= torch.zeros(1).cuda(cuda_device)
                        #title_pos_mean_sum= torch.zeros(1).cuda(cuda_device)
                        #title_neg_mean_sum = torch.zeros(1).cuda(cuda_device)
                        #inter_pod_diff_sum = torch.zeros(1).cuda(cuda_device)

                        grad_norm_sum = torch.zeros(1).cuda(cuda_device)
                        grad_max_sum = torch.zeros(1).cuda(cuda_device)
                        grad_counter = 0
                        # make sure that the perf of the queue is sustained
                        if training_queue.qsize() < 10:
                            logger.warning("training_queue.qsize() < 10")

                        lr_scheduler.step()
                        tb_writer.add_scalar('Learning Rate', lr_scheduler.get_lr()[0], global_update_i)
                        #print(lr_scheduler.get_lr())

                        #tb_writer.add_scalar('MLM/Total tokens', sum_tokens, global_i)
                        #tb_writer.add_scalar('MLM/% Masked', sum(token_masked_counter.values())/sum_tokens, global_i)

                        ## token distribution figure
                        #if i % 10_000==0:
                        #    fig = plt.figure(figsize=(7.8,7.8))
                        #    ax = fig.add_subplot(1, 1, 1)
                        #    sorted_list=sorted(list(token_masked_counter.values()),reverse=True)
                        #    ax.plot(range(0,len(token_masked_counter.values())),sorted_list)
                        #    ax.set_xscale("log")
                        #    ax.set_yscale("log")
                        #    tb_writer.add_figure("MLM/Token Distribution",fig,global_i)
                        #    #plt.savefig("test.png")
                        #
                        #    #token_masked_counter = defaultdict(int)
                        #    #sum_tokens = 0

                except RuntimeError as r:
                    if r.args[0].startswith("CUDA out of memory"): # lol yeah that is python for you 
                        if i - lastRetryI < 2:
                            raise r

                        del loss,batch
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
                if i > 0 and i % save_model_every == 0:
                    if is_distributed:
                        logger.info(str(global_i)+": "+model.module.get_param_stats())
                        torch.save(model.module.neural_ir_model.state_dict(), best_model_store_path)
                    else:
                        logger.info(str(global_i)+": "+model.get_param_stats())
                        torch.save(model.neural_ir_model.state_dict(), best_model_store_path)

                progress.update()
                i+=1
                global_i+=1

            progress.close()
            # make sure we didn't make a mistake in the configuration / data preparation
            if training_queue.qsize() != 0:
                logger.error("training_queue.qsize() is not empty after epoch "+str(epoch))

            if is_distributed:
                torch.save(model.module.neural_ir_model.state_dict(), os.path.join(run_folder, "model_epoch_"+str(epoch)+".pytorch-state-dict"))
            else:
                torch.save(model.neural_ir_model.state_dict(), os.path.join(run_folder, "model_epoch_"+str(epoch)+".pytorch-state-dict"))
            

            train_exit.set()  # allow sub-processes to exit
            perf_monitor.stop_block("train",i - perf_start_inst)
       
        perf_monitor.log_value("eval_gpu_mem",str(torch.cuda.memory_allocated()/float(1e9)) + " GB")
        perf_monitor.log_value("eval_gpu_mem_max",str(torch.cuda.max_memory_allocated()/float(1e9)) + " GB")

        perf_monitor.save_summary(os.path.join(run_folder,"perf-monitor.txt"))
        tb_writer.close()

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
