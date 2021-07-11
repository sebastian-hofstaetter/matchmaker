#
# intersect a query (or any id starting .tsv file) with a qrel file (useful to reduce set of queries)
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
                    help='the query output file location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

parser.add_argument('--train-ids', action='store', dest='train_file',
                    help='the train triples file', required=True)

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

train_queries = set()
with open(args.train_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        train_queries.add(ls[2])

#
# produce output
# -------------------------------
#  
with open(args.out_file,"w",encoding="utf8") as out_file:

    for q_id,q_line in tqdm(queries.items()):
        if q_id not in train_queries:
            out_file.write(q_line)
