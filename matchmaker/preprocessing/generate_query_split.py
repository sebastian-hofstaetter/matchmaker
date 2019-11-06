#
# generate a query split for msmarco 
# - take the full query set (queries.dev/eval) and the published subset (top1000.dev/eval) 
#  -> produce a query file containing all queries not in the subset  
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from matchmaker.dataloaders.ir_labeled_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file-subset', action='store', dest='out_file_sub',
                    help='the query output file location', required=True)

parser.add_argument('--out-file-not-sub', action='store', dest='out_file_notsub',
                    help='the query output file location', required=True)


parser.add_argument('--subset-val-file', action='store', dest='subset_file',
                    help='top1000.<x>.tsv location', required=True)

parser.add_argument('--full-query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only output judged queries', required=False)

args = parser.parse_args()


#
# load data 
# -------------------------------
#
if args.qrel:
    qrels = load_reference(args.qrel)

subset = {} # int id -> full line dictionary
with open(args.subset_file,"r",encoding="utf8") as subset_file:
    for line in tqdm(subset_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        subset[_id] = None

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        queries[_id] = ls[1].rstrip()

#
# produce output
# -------------------------------
#  
with open(args.out_file_sub,"w",encoding="utf8") as sub_out_file:
    with open(args.out_file_notsub,"w",encoding="utf8") as notsub_out_file:

        for q in tqdm(queries):
            if not args.qrel or q in qrels:
                if q in subset:
                    sub_out_file.write("\t".join([str(q),queries[q]])+"\n")
                else:
                    notsub_out_file.write("\t".join([str(q),queries[q]])+"\n")
