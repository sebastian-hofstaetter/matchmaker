#
# cluster queries from a given dense retrieval model
# -------------------------------


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

import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
import numpy
import random

from allennlp.nn.util import move_to_device
from matchmaker.utils.utils import *
from matchmaker.utils.config import *

from matchmaker.models.all import get_model, get_word_embedder, build_model
from matchmaker.modules.indexing_heads import *

from typing import Dict, Tuple, List
from matchmaker.utils.input_pipeline import *
from matchmaker.utils.performance_monitor import * 
from matchmaker.eval import *
from matchmaker.utils.core_metrics import *
from matchmaker.retrieval.faiss_indices import *

Tqdm.default_mininterval = 1

if __name__ == "__main__":


    #
    # config
    # -------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', action='store', dest='run_name',
                        help='run name, used for the run folder (no spaces, special characters)', required=False)
    parser.add_argument('--config-file', nargs='+', action='store', dest='config_file',
                        help='config file with all hyper-params & paths', required=False)
    parser.add_argument('--config-overwrites', action='store', dest='config_overwrites',
                        help='overwrite config values -> key1: valueA,key2: valueB ', required=False)
    parser.add_argument('--continue-folder', action='store', dest='continue_folder',
                        help='path to experiment folder, which should be continued', required=False)


    args = parser.parse_args()

    if args.continue_folder:
        from_scratch = False
        run_folder = args.continue_folder
        config = get_config_single(os.path.join(run_folder, "config.yaml"), args.config_overwrites)
    else:
        if not args.run_name:
            raise Exception("--run-name must be set (or continue-folder)")
        from_scratch = True
        config = get_config(args.config_file, args.config_overwrites)
        run_folder = prepare_experiment(args, config)

    logger = get_logger_to_file(run_folder, "main")

    logger.info("Running: %s", str(sys.argv))
    model_config = get_config_single(os.path.join(config["trained_model"], "config.yaml"))
    model_config["batch_size_eval"] = config["inference_batch_size"] # overwrite original batch size

    #
    # random seeds
    #
    torch.manual_seed(model_config["random_seed"])
    numpy.random.seed(model_config["random_seed"])
    random.seed(model_config["random_seed"])

    logger.info("Torch seed: %i ",torch.initial_seed())

    # hardcode gpu usage
    cuda_device = 0 # always take the first -> set others via cuda flag in bash
    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("startup")
    
    #
    # create and load model instance
    # -------------------------------
    
    word_embedder, padding_idx = get_word_embedder(model_config)
    model, encoder_type = get_model(model_config,word_embedder,padding_idx)
    model = build_model(model,encoder_type,word_embedder,model_config)

    model_path = os.path.join(config["trained_model"], "best-model.pytorch-state-dict")
    load_result = model.load_state_dict(torch.load(model_path),strict=False)
    logger.info('Warmstart init model from:  %s', model_path)
    logger.info(load_result)
    print("Warmstart Result:",load_result)

    #model_indexer = CollectionIndexerHead(model).cuda()
    model_searcher = QuerySearcherHead(model).cuda()

    logger.info('Model %s total parameters: %s', model_config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)

    is_distributed = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #model_indexer = torch.nn.DataParallel(model_indexer)
        model_searcher = torch.nn.DataParallel(model_searcher)
        is_distributed = True
    perf_monitor.set_gpu_info(torch.cuda.device_count(),torch.cuda.get_device_name())


    perf_monitor.stop_block("startup")
    perf_monitor.start_block("inference")
    #
    # run through the data
    # -------------------------------

    token_base_size = config["token_block_size"]
    token_dimensions = config["token_dim"]

    try:
        if from_scratch:
            seq_ids = []
            id_mapping = []
            storage = []
        
            model_searcher.eval()
            input_loader = allennlp_single_sequence_loader(model_config,config, config["index_queries_tsv"], sequence_type="query")
            with torch.no_grad():
                progress = Tqdm.tqdm()
                i=0
                for batch_orig in input_loader:

                    batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)

                    output = model_searcher.forward(batch["seq_tokens"], use_fp16=model_config["use_fp16"])
                    output = output.cpu().numpy()  # get the output back to the cpu - in one piece

                    for sample_i, seq_id in enumerate(batch_orig["seq_id"]):
                        id_mapping.append(len(seq_ids))
                        seq_ids.append(seq_id)
                    storage.append(output)

                    progress.update()
                    i+=1
                progress.close()

            # save last token reps
            storage = numpy.concatenate(storage,axis=0)
            id_mapping = numpy.array(id_mapping,dtype=numpy.int64)

            import zipfile
            def saveCompressed(fh, **namedict):
                with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_STORED,
                                     allowZip64=True) as zf:
                    for k, v in namedict.items():
                        with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                            numpy.lib.npyio.format.write_array(buf,
                                                               numpy.asanyarray(v),
                                                               allow_pickle=True)

            saveCompressed(os.path.join(run_folder,"query_vectors_n_ids.npz"),storage=storage,id_mapping=id_mapping,seq_ids=seq_ids)

            perf_monitor.log_value("eval_gpu_mem",str(torch.cuda.memory_allocated()/float(1e9)) + " GB")
            perf_monitor.log_value("eval_gpu_mem_max",str(torch.cuda.max_memory_allocated()/float(1e9)) + " GB")
            perf_monitor.stop_block("inference")

        else:
            dfs = numpy.load(os.path.join(run_folder,"query_vectors_n_ids.npz"))
            storage=dfs.get("storage")[()]
            id_mapping=dfs.get("id_mapping")[()]
            seq_ids=dfs.get("seq_ids")[()]
                
        #
        # nearest neighbor indexing
        # -------------------------
        perf_monitor.start_block("indexing")
        
        indexer = FaissDynamicIndexer(config)

        indexer.prepare([storage])
        indexer.index_all([id_mapping], [storage])

        perf_monitor.stop_block("indexing")

        #
        # cluster info output
        # -------------------------
        perf_monitor.start_block("output")
        
        id_text={}
        with open(config["cluster_queries_tsv"],"r",encoding="utf8") as qf:
            for l in qf:
                l=l.split("\t")
                id_text[l[0]] = l[1].strip()

        clusters = [[] for _ in range(config["faiss_ivf_list_count"])]

        input_loader = allennlp_single_sequence_loader(model_config,config, config["cluster_queries_tsv"], sequence_type="query")
        with torch.no_grad():
            progress = Tqdm.tqdm()
            i=0
            for batch_orig in input_loader:

                batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)

                output = model_searcher.forward(batch["seq_tokens"], use_fp16=model_config["use_fp16"])
                output = output.cpu().numpy()  # get the output back to the cpu - in one piece

                for sample_i, seq_id in enumerate(batch_orig["seq_id"]):
                    _, _, centroid_ids = indexer.search_single(output[sample_i],1)

                    clusters[int(centroid_ids)].append(seq_id)

                #    id_mapping.append(len(seq_ids))
                #    seq_ids.append(seq_id)
                #storage.append(output)

                progress.update()
                i+=1
            progress.close()


        #clusters = indexer.get_all_cluster_assignments()

        with open(os.path.join(run_folder,"cluster-assignment-ids.tsv"),"w",encoding="utf8") as out_file,\
             open(os.path.join(run_folder,"cluster-assignment-text.tsv"),"w",encoding="utf8") as out_file_text:
            for clust in clusters:
                out_file.write("\t".join(idx for idx in clust)+"\n")
                out_file_text.write("\n".join(idx+"\t"+id_text[idx] for idx in clust)+\
                                    "\n--------------------------------------------\n")

        perf_monitor.stop_block("output")
        perf_monitor.save_summary(os.path.join(run_folder,"perf-monitor.txt"))

    except:
        logger.info('-' * 89)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        print("----- Attention! - something went wrong in the train loop (see logger) ----- ")

    exit(0)