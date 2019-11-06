#
# generates hybrid results from two orignal outputs -> based on 2 query splits (output from oov_words_finder.py)
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import statistics

from matchmaker.evaluation.msmarco_eval import *

from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from msmarco_eval import *

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--res-1', action='store', dest='res_1',
                    help='first result', required=True)

parser.add_argument('--res-2', action='store', dest='res_2',
                    help='second result', required=True)

parser.add_argument('--query-1', action='store', dest='query_1',
                    help='keep queries for 1 result', required=True)

parser.add_argument('--query-2', action='store', dest='query_2',
                    help='keep queries for 2 result', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to evaluate hybrid results', required=True)

parser.add_argument('--out', action='store', dest='out',
                    help='out file for hybrid results', required=True)

args = parser.parse_args()


#
# generate hybrid
# -------------------------------
# 

query_1_ids = {}
query_2_ids = {}

with open(args.query_1,"r",encoding="utf8") as query_1:
    for l in query_1:
        query_1_ids[l.split()[0]] = None

with open(args.query_2,"r",encoding="utf8") as query_2:
    for l in query_1:
        query_2_ids[l.split()[0]] = None

with open(args.out,"w",encoding="utf8") as out:

    with open(args.res_1,"r",encoding="utf8") as res_1:
        for l in res_1:
            if l.split()[0] in query_1_ids:
                out.write(l)

    with open(args.res_2,"r",encoding="utf8") as res_2:
        for l in res_2:
            if l.split()[0] in query_2_ids:
                out.write(l)

#
# evaluate
# -------------------------------
#  

metrics = compute_metrics_from_files(args.qrel,args.out)

print(metrics)
