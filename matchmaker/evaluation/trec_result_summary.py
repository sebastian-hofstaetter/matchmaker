#
# take a bunch of trec_eval result files (-q all queries) and make a summary table out of it
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import statistics

from matchmaker.utils import *

from matchmaker.evaluation.msmarco_eval import *
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

import glob
#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--res-in', action='store', dest='res_in',
                    help='result file glob in', required=True)

parser.add_argument('--out', action='store', dest='out',
                    help='summary table out ', required=True)
                    
args = parser.parse_args()


#
# compare (different mrr gains up,same,down)
# -------------------------------
#  

res_files = glob.glob(args.res_in)

data = {}
file_names=[]

summary_measures = ["map","ndcg","P_10","recall_1000"]

for file in res_files:
    file_name = os.path.basename(file)
    file_names.append(file_name)
    with open(file,"r",encoding="utf8") as open_file:
        for l in open_file:
            l = l.split()
            measure = l[0]
            qid = l[1]
            value = l[2]

            if qid not in data:
                data[qid] = {}

            if file_name not in data[qid]:
                data[qid][file_name] = {}

            data[qid][file_name][measure] = value


with open(args.out,"w",encoding="utf8") as out_file:
    out_file.write("sep=,\n")
    for file_name in file_names:
        out_file.write(","+file_name+",,,")
    out_file.write("\nTopics")
    for file_name in file_names:
        for m in summary_measures:
            out_file.write(","+m)
    out_file.write("\n")

    for qid,q_data in data.items():

        out_file.write(qid)
        for file_name in file_names:
            for m in summary_measures:
                out_file.write(","+data[qid][file_name][m])
        out_file.write("\n")
