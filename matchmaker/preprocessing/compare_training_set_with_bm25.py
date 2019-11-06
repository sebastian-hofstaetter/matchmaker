#
# generate validation.tsv tuples from candidate set files 
# (for example lucene trec output for bm25 et al.)
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

parser.add_argument('--out-file', action='store', dest='val_out_file',
                    help='validation output file location', required=True)

parser.add_argument('--candidate-file', action='store', dest='candidate_file',
                    help='trec ranking file location (lucene output)', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

parser.add_argument('--top-N', action='store', dest='top_n',type=int,
                    help='how many docs per query to take', required=True)


args = parser.parse_args()


#
# load data 
# -------------------------------
# 
collection = {} # int id -> full line dictionary
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        collection[_id] = ls[1].rstrip()

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
max_rank = args.top_n

with open(args.val_out_file,"w",encoding="utf8") as val_out_file:
    with open(args.candidate_file,"r",encoding="utf8") as candidate_file:

        for line in tqdm(candidate_file):
            ls = line.split() # 2 Q0 1782337 1 21.656799 Anserini

            if int(ls[3]) > max_rank:
                continue

            query_id = int(ls[0])
            doc_id = int(ls[2])
            
            if query_id in queries:
                val_out_file.write("\t".join([ls[0],ls[2],queries[query_id],collection[doc_id]])+"\n")