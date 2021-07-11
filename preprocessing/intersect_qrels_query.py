#
# intersect a query (or any id starting .tsv file) with a qrel file (useful to reduce set of qrels)
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='the qrel output file location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only output judged queries', required=True)

args = parser.parse_args()


#
# load data 
# -------------------------------
#

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        queries[ls[0]] = line

#
# produce output
# -------------------------------
#  
with open(args.qrel,"r",encoding="utf8") as qrel_file, \
     open(args.out_file,"w",encoding="utf8") as out_file:

    for q_line in tqdm(qrel_file):
        q_id = q_line.split()[0]
        if q_id in queries:
            out_file.write(q_line)
