import yaml
from timeit import default_timer
from typing import Dict,List
from datetime import datetime
import os
import logging
import argparse
import shutil
import numpy
from rich.console import Console
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
from matchmaker.utils.config import save_config
from rich.console import Console
console = Console()

class Timer():
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = default_timer()
        print(self.message + " started ...")

    def __exit__(self, type, value, traceback):
        print(self.message+" finished, after (s): ",
              (default_timer() - self.start_time))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', action='store', dest='run_name',
                        help='run name, used for the run folder (no spaces, special characters)', required=False)
    parser.add_argument('--run-folder', action='store', dest='run_folder',
                        help='run folder if it exists, if not set a new one is created using run-name', required=False)

    parser.add_argument('--config-file', nargs='+', action='store', dest='config_file',
                        help='config file with all hyper-params & paths', required=False)

    parser.add_argument('--continue-folder', action='store', dest='continue_folder',
                        help='path to experiment folder, which should be continued', required=False)

    parser.add_argument('--config-overwrites', action='store', dest='config_overwrites',
                        help='overwrite config values -> key1: valueA,key2: valueB ', required=False)

    return parser

def get_logger_to_file(run_folder,name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger.setLevel(logging.INFO)

    log_filepath = os.path.join(run_folder, 'log.txt')
    file_hdlr = logging.FileHandler(log_filepath)
    file_hdlr.setFormatter(formatter)
    file_hdlr.setLevel(logging.INFO)
    logger.addHandler(file_hdlr)

    return logger

def prepare_experiment_folder(base_path, run_name):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H%M')

    run_folder = os.path.join(base_path,time_stamp + "_" + run_name)

    os.makedirs(run_folder)

    return run_folder

def prepare_experiment(args, config):

    run_folder = prepare_experiment_folder(config["expirement_base_path"], args.run_name)
    #
    # saved uased config (with overwrites)
    #     
    save_config(os.path.join(run_folder,"config.yaml"),config)

    #
    # copy source code of matchmaker
    #
    dir_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    shutil.copytree(dir_path, os.path.join(run_folder,"matchmaker-src"), ignore=shutil.ignore_patterns("__pycache__"))

    return run_folder

def parse_candidate_set(file_path, to_N):
    candidate_set = {} # dict[qid] -> dict[did] -> rank

    with open(file_path, "r") as cs_file:
        for line in cs_file:
            line = line.split()
            
            rank = int(line[3])

            if rank <= to_N:

                q_id = line[0]
                d_id = line[2]

                if q_id not in candidate_set:
                    candidate_set[q_id] = {}
                candidate_set[q_id][d_id] = rank

    return candidate_set

def print_hello(config:dict,run_folder:str,train_mode,show_settings=["Model","Train Batch Size","Train Learning Rate","Max Doc. Length","Init"]):

    embedding_tech = config["token_embedder_type"] if not config["token_embedder_type"].startswith("bert") else config["bert_pretrained_model"]
    settings = {

        "Model": config["model"] +" ("+embedding_tech+")",

        #
        # training settings
        #
        "Train Batch Size": str(config.get("batch_size_train")) + " ("+ ("No gradient acc." if config.get("gradient_accumulation_steps",-1) == -1 else str(config.get("gradient_accumulation_steps"))+" Gradient acc. steps") + ")",
        "Train Learning Rate":( str(config.get("pretrain_lr")) if train_mode=="Pre-Train" else str(config.get("param_group0_learning_rate"))),
        "Max Doc. Length": str(config["max_doc_length"]),
        "Init": ("Training from scratch" if "warmstart_model_path" not in config else "Checkpoint " + os.path.basename(os.path.dirname(config["warmstart_model_path"]))),

        #
        # dense retrieval
        #
        "Trained Checkpoint": config.get("trained_model",""),
        "Index": config.get("faiss_index_type","") + (" (On GPU)" if config.get("faiss_use_gpu",False) and config.get("faiss_index_type","") != "hnsw" else "") +\
                " | "+ ", ".join([v +": " +str(k) for v,k in config.items() if config.get("faiss_index_type","no_index") in v]),
        "Collection Batch Size": str(config.get("collection_batch_size","")),
        "Query Batch Size": str(config.get("query_batch_size","")),
        "Use ONNX Runtime": str(config.get("onnx_use_inference",False)),
    }

    logo = Text.assemble(\
    ("\n         (@@&@@&&@%                  \n" +\
    "      &@@.         @@@.                \n" +\
    "    &@#  / .         .@&               \n" +\
    "   @@ .@&              @@              \n" +\
    "   @@ &@               &@/             \n" +\
    "   @& &@               @&              \n" +\
    "    @&                @&*              \n" +\
    "     %@&/          .@&@@               \n" +\
    "        #@@@@@@&&@@@ /@@@@ @           \n" +\
    "                      *@@@%@/&,        \n" +\
    "                         @@@&@@ @      \n" +\
    "                           ,@&@&@&@.   \n" +\
    "                              @&@@@    \n","gray"))

    
    table = Table(show_header=True, header_style="bold magenta")
    table.title=""
    table.add_column("Configuration")
    table.add_column("")
    table.box = box.SIMPLE_HEAD

    for s in show_settings:
        table.add_row(s,settings[s])

    layout = Layout()
    layout.split_row(
        Layout(name="logo",size=40),
        Layout(name="body"),
    )
    layout["logo"].update(logo)
    layout["body"].update(Align.left(table))

    console = Console(height=15)
    console.print(Text.assemble(("\n   Matchmaker","deep_pink4"), ("   "+train_mode+"\n","magenta"),justify="left"))
    console.print(layout)
    console.print(Text("Experiment: " + run_folder + "\n"))

def read_best_info(path):

    #sep=,
    #Epoch,batch_number,cs@n,MRR,QueriesRanked,QueriesWithNoRelevant,QueriesWithRelevant,AverageRankGoldLabel@10,MedianRankGoldLabel@10,AverageRankGoldLabel,MedianRankGoldLabel,HarmonicMeanRankingGoldLabel
    #0,28000,95,0.2271499636148629,6980,3741,3239,3.5723988885458473,3,1.6577363896848138,0.0,0

    with open(path, "r") as bi_file:
        next(bi_file) # igonre: sep=,
        headers = next(bi_file).split(",")
        values = next(bi_file).split(",")

    best_metric_info = {}
    best_metric_info["metrics"]={}
    best_metric_info["metrics"]["MRR@10"] = float(values[headers.index("MRR@10")])
    best_metric_info["epoch"] = int(values[headers.index("Epoch")])
    best_metric_info["batch_number"] = int(values[headers.index("batch_number")])
    
    best_metric_info["cs@n"] = "-"
    if values[headers.index("cs@n")] != "-":
        best_metric_info["cs@n"] = int(values[headers.index("cs@n")])

    return best_metric_info

import zipfile
def saveCompressed(fh, **namedict):
    with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for k, v in namedict.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                numpy.lib.npyio.format.write_array(buf,
                                                   numpy.asanyarray(v),
                                                   allow_pickle=True)


#
# from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
# Thanks!
#
class EarlyStopping():
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        self.stop = False
        #if patience == 0:
        #    self.is_better = lambda a, b: True
        #    self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if numpy.isnan(metrics):
            self.stop = True
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.stop = True
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)