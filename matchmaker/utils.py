import yaml
from timeit import default_timer
from typing import Dict,List
from datetime import datetime
import os
import logging
import argparse
import shutil
import numpy

class Timer():
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = default_timer()
        print(self.message + " started ...")

    def __exit__(self, type, value, traceback):
        print(self.message+" finished, after (s): ",
              (default_timer() - self.start_time))

def get_config(config_path:List[str], overwrites:str = None) ->Dict[str, any] :
    cfg = {}
    for path in config_path:
        with open(os.path.join(os.getcwd(), path), 'r') as ymlfile:
            cfg.update(yaml.load(ymlfile))

    if overwrites is not None and overwrites != "":
        over_parts = [yaml.load(x) for x in overwrites.split(",")]
        
        for d in over_parts:
            for key, value in d.items():
                cfg[key] = value

    return cfg

def get_config_single(config_path:str, overwrites:str = None) ->Dict[str, any] :
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    if overwrites is not None and overwrites != "":
        over_parts = [yaml.load(x) for x in overwrites.split(",")]
        
        for d in over_parts:
            for key, value in d.items():
                cfg[key] = value

    return cfg

def save_config(config_path:str, config:Dict[str, any]):
    with open(config_path, 'w') as ymlfile:
        yaml.safe_dump(config, ymlfile,default_flow_style=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', action='store', dest='run_name',
                        help='run name, used for the run folder (no spaces, special characters)', required=True)
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
    #if args.run_folder is not None:
    #    run_folder = args.run_folder
    #else:
    run_folder = prepare_experiment_folder(config["expirement_base_path"], args.run_name)
    #
    # saved uased config (with overwrites)
    #     
    save_config(os.path.join(run_folder,"config.yaml"),config)

    #
    # copy source code of matchmaker
    #
    dir_path = os.path.dirname(os.path.realpath(__file__))

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
    best_metric_info["cs@n"] = int(values[headers.index("cs@n")])

    return best_metric_info

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