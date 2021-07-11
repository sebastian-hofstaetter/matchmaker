#
# gather results from multiple model runs for an ensemble
# -------------------------------
#
# - needs a set of finished runs from train.py
#
#
import argparse
import copy
import os

import glob
import time
import sys
sys.path.append(os.getcwd())
import numpy
import statistics
from matchmaker.utils.utils import *
from matchmaker.eval import *
from matchmaker.core_metrics import *

parser = argparse.ArgumentParser()

parser.add_argument('--ensemble-folder', action='store', dest='ensemble_folder',
                    help='main folder for result locations', required=True)
parser.add_argument('--ensemble-type', action='store', dest='ensemble_type',
                    help='avg or rrf (reciprocal rank fusion)', required=True)

parser.add_argument('--run-config', action='store', dest='config_file',
                    help='config file with all hyper-params & paths', required=True)
parser.add_argument('--config-overwrites', action='store', dest='config_overwrites',
                    help='overwrite config values -> key1: valueA,key2: valueB ', required=False)

parser.add_argument('--runs', action='store', dest='run_locations',
                    help='path to run folders, which should be combined to the ensemble, concatenated with "," ', required=True)

args = parser.parse_args()

config = get_config_single(os.path.join(os.getcwd(), args.config_file), args.config_overwrites)
os.makedirs(args.ensemble_folder,exist_ok=True)
logger = get_logger_to_file(args.ensemble_folder, "main")

use_rrf = args.ensemble_type == "rrf"
rrf_k = 60

#run_folders = glob.glob(args.run_locations) 
run_folders = args.run_locations.split(",")

print("run_folders:",run_folders)

#
# create stats for validation/test metrics
#

def load_results(file_name):
    all_results = {}
    for folder in run_folders:
        with open(os.path.join(folder,file_name),"r") as r_file:
            results_dict = {}
            next(r_file) # ignores sep=,
            line1 = next(r_file).split(",")
            line2 = next(r_file).split(",")
            for i,header in enumerate(line1):
                header = header.strip()
                data = line2[i]
                results_dict[header] = data
            all_results[folder] = results_dict
    return all_results

def create_save_stats(raw_data,out_prefix):
    sum_dict = {}
    best_per_folder = {}

    for folder, res in raw_data.items():
        for header, value in res.items():
            if header not in sum_dict:
                sum_dict[header] = []
            sum_dict[header].append(float(value))
            if header == "nDCG@10":
                best_per_folder[folder] = float(value)

    with open(os.path.join(args.ensemble_folder,out_prefix+"-stats.csv"),"w") as out_file:
        out_file.write("sep=,\nMetric, Min, Max, Mean, Median, Stdev, Variance\n")

        for header, value_list in sum_dict.items():
            min_value = min(value_list)
            max_value = max(value_list)
            mean = statistics.mean(value_list)
            median = statistics.median(value_list)
            variance = statistics.variance(value_list)
            stdev=statistics.stdev(value_list)

            out_file.write(",".join([str(header),str(min_value),str(max_value),str(mean),str(median),str(stdev),str(variance)])+"\n")
    
    return best_per_folder

# todo gather stats
#best_infos_per_folder_validation = create_save_stats(load_results("dev_top1000-metrics.csv"), "dev")
#create_save_stats(load_results("trec2019_top1000-metrics.csv"), "trec2019_top1000")
#create_save_stats(load_results("trec2020_top1000-metrics.csv"), "trec2020_top1000")

#
# create ensemble result from raw results for validation/test
#
def load_candidate_from_stream_with_score(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split()
            qid = l[0]
            pid = l[1].strip()
            rank = int(l[2])
            score = float(l[3])
            if qid not in qid_to_ranked_candidate_passages:
                qid_to_ranked_candidate_passages[qid] = []
            qid_to_ranked_candidate_passages[qid].append((pid,rank,score))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages

def find_best_ensemble(best_infos,in_file,qrel_file,out_prefix,binarization_point=1):
    
    best_result_folders = []

    sorted_folders = [(k, best_infos[k]) for k in sorted(best_infos, key=best_infos.get, reverse=True)]
    print("sorted_folders",sorted_folders)

    all_results = {}
    for folder in run_folders:
        r_file = os.path.join(folder, in_file)
        with open(r_file,"r") as r_file:
            all_results[folder] = load_candidate_from_stream_with_score(r_file)

    best_mrr_so_far = 0

    for sorted_folder,_ in sorted_folders:

        current_folders = best_result_folders + [sorted_folder]
        # todo try out more combinations, based on above gathered stats !
        merged_results = {}

        for folder, res in all_results.items():
            if folder in current_folders:
                for query, passages_scores in res.items():
                    if query not in merged_results:
                        merged_results[query] = {}
                    for passage, rank, score in passages_scores:
                        if passage not in merged_results[query]:
                            merged_results[query][passage] = []
                        if use_rrf:
                            merged_results[query][passage].append(rank)
                        else:
                            merged_results[query][passage].append(score)

        ensemble_results = {}

        for query, passages in merged_results.items():
            ensemble_results[query] = []
            if use_rrf:
                for passage in passages:
                    rr = [1 / (rrf_k + rank) for rank in merged_results[query][passage]]
                    ensemble_results[query].append((passage, sum(rr)))
            else:
                for passage in passages:
                    ensemble_results[query].append((passage,statistics.mean(merged_results[query][passage]))) # mean is BETTTTTER than median

        ensemble_file = os.path.join(args.ensemble_folder, out_prefix + "-ensemble-output.txt")
        save_sorted_results(ensemble_results,ensemble_file)

        #if qrel_file != None:
            # todo do cs@n computation here -> also allow as param for test & leaderboard

        metrics = calculate_metrics_plain(load_ranking(ensemble_file), load_qrels(qrel_file),binarization_point=binarization_point)
        print("nDCG@10:",metrics["nDCG@10"],current_folders)

        if metrics["nDCG@10"] > best_mrr_so_far:
            best_result_folders = current_folders
            best_mrr_so_far = metrics["nDCG@10"]
            print("got new best")
            metric_file_path = os.path.join(args.ensemble_folder, out_prefix + "-ensemble-metrics.csv")
            save_fullmetrics_oneN(metric_file_path,metrics,-1,-1)

    return best_result_folders


def create_save_ensemble(best_run_folders,in_file,qrel_file,out_prefix,binarization_point=1):
    
    all_results = {}
    for folder in best_run_folders:
        r_file = os.path.join(folder, in_file)
        with open(r_file,"r") as r_file:
            all_results[folder] = load_candidate_from_stream_with_score(r_file)

    # todo try out more combinations, based on above gathered stats !
    merged_results = {}

    for folder, res in all_results.items():
        for query, passages_scores in res.items():
            if query not in merged_results:
                merged_results[query] = {}
            for passage, rank, score in passages_scores:
                if passage not in merged_results[query]:
                    merged_results[query][passage] = []
                if use_rrf:
                    merged_results[query][passage].append(rank)
                else:
                    merged_results[query][passage].append(score)

    ensemble_results = {}

    for query, passages in merged_results.items():
        ensemble_results[query] = []
        if use_rrf:
            for passage in passages:
                rr = [1 / (rrf_k + rank) for rank in merged_results[query][passage]]
                ensemble_results[query].append((passage, sum(rr)))
        else:
            for passage in passages:
                ensemble_results[query].append((passage,statistics.mean(merged_results[query][passage]))) # mean is BETTTTTER than median

    ensemble_file = os.path.join(args.ensemble_folder, out_prefix + "-ensemble-output.txt")
    save_sorted_results(ensemble_results,ensemble_file)

    if qrel_file != None:
        # todo do cs@n computation here -> also allow as param for test & leaderboard

        metrics = calculate_metrics_plain(load_ranking(ensemble_file), load_qrels(qrel_file),binarization_point=binarization_point)

        metric_file_path = os.path.join(args.ensemble_folder, out_prefix + "-ensemble-metrics.csv")
        save_fullmetrics_oneN(metric_file_path,metrics,-1,-1)


#best_folders = find_best_ensemble(best_infos_per_folder_validation,"test_ridiculous_T2ensemble_dev-output.txt",config["test"]["ridiculous_T2ensemble_dev"]["qrels"],"ensemble_dev")
#best_folders = find_best_ensemble(best_infos_per_folder_validation,"dev_top1000-output.txt",config["query_sets"]["dev_top1000"]["qrels"],"dev_top1000")
#print("----\nbest_folders:",best_folders)

best_folders = run_folders

#
# re-ranking result categories
#
if "validation_end" in config:
    for label,info in config["validation_end"].items():
        create_save_ensemble(best_folders,"val-"+label+"-output.txt",info["qrels"],"val-"+label,info["binarization_point"])

if "test" in config:
    for label,info in config["test"].items():
        create_save_ensemble(best_folders,"test_"+label+"-output.txt",info["qrels"],"test_"+label,info["binarization_point"])

if "leaderboard" in config:
    for label,info in config["leaderboard"].items():
        create_save_ensemble(best_folders,"leaderboard"+label+"-output.txt",None,"leaderboard"+label) # here we have no qrels 

#
# retrieval result categories
#
if "query_sets" in config:
    for label,info in config["query_sets"].items():
        if "qrels" in info:
            create_save_ensemble(best_folders,label+"-output.txt",info["qrels"],label,info["binarization_point"])
        else:
            create_save_ensemble(best_folders,label+"-output.txt",None,label) # here we have no qrels 