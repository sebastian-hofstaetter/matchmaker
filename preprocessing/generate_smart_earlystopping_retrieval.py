#
# generate validation.tsv tuples from qrel files 
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from blingfire import *
import random
random.seed(208973249)

from matchmaker.core_metrics import load_qrels
import numpy

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='val_out_file',
                    help='validation output file location', required=True)

parser.add_argument('--candidate-metric', action='store', dest='candidate_metric',
                    help='qrels', required=True)

parser.add_argument('--candidate-file', action='store', dest='candidate_file',
                    help='qrels', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrels', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

args = parser.parse_args()

max_doc_char_length = 100_000
number_of_sampled_queries = 4_000
max_rank = 100


#
# load data 
# -------------------------------
# 
qrels = load_qrels(args.qrel)

collection = {} # int id -> full line dictionary
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        collection[_id] = ls[1].rstrip()[:max_doc_char_length]

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        queries[_id] = ls[1].rstrip()

query_metrics = {}
with open(args.candidate_metric,"r",encoding="utf8") as candidate_metric:
    for line in tqdm(candidate_metric):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        query_metrics[_id] = float(ls[1].rstrip())


#
# produce output
# -------------------------------
#  

query_metrics_only = numpy.array(list(query_metrics.values()))

indices = numpy.digitize(query_metrics_only,numpy.arange(numpy.min(query_metrics_only),
                                                         numpy.max(query_metrics_only),
                                                         (numpy.max(query_metrics_only)-numpy.min(query_metrics_only))/5))
query_id_bins = [[] for _ in range(5)]
for i,(q_id,q_m) in enumerate(query_metrics.items()):
    query_id_bins[indices[i]-1].append(q_id)

sampled_query_ids = []

no_to_sample = number_of_sampled_queries // len(query_id_bins)
for bin in query_id_bins:
    sampled_query_ids.extend(random.sample(bin,min(len(bin),no_to_sample)))

sampled_query_ids = set(sampled_query_ids)
print("got",len(sampled_query_ids),"queries")


known_pairs = set()

with open(args.val_out_file,"w",encoding="utf8") as val_out_file:
    with open(args.candidate_file,"r",encoding="utf8") as candidate_file:

        for line in tqdm(candidate_file):
            ls = line.split() # 2 Q0 1782337 1 

            query_id = ls[0]
            doc_id = ls[1]

            if query_id not in sampled_query_ids:
                continue

            if int(ls[2]) > max_rank:
                continue

            if (query_id,doc_id) not in known_pairs:
                known_pairs.add((query_id,doc_id))

            out_arr = [query_id,doc_id,queries[query_id],collection[doc_id]]
            val_out_file.write("\t".join(out_arr)+"\n")

    for q_id in sampled_query_ids:
        for rel_doc in qrels[q_id]:
            if (q_id,rel_doc) not in known_pairs:
                
                out_arr = [q_id,rel_doc,queries[q_id],collection[rel_doc]]
                val_out_file.write("\t".join(out_arr)+"\n")
