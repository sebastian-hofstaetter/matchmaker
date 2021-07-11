#
# msmarco doc: create the train.tsv triples  
# -------------------------------

import random
random.seed(42)

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *
from collections import defaultdict
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='training output text file location', required=True)

parser.add_argument('--in-file-ids', action='store', dest='in_file_ids',
                    help='ids file location', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

args = parser.parse_args()

max_doc_char_length = 150_000
max_doc_token_length = 10000

#
# load data 
# -------------------------------
# 
collection = {}
#collection_length = {} 
tokenizer = BlingFireTokenizer()
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        max_char_doc = ls[1].rstrip()[:max_doc_char_length]
        max_char_doc2 = ls[2].rstrip()[:max_doc_char_length]
        collection[_id] = max_char_doc + "\t" + max_char_doc2
        #collection_length[_id] = len(tokenizer.tokenize(max_char_doc))

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        queries[_id] = ls[1].rstrip()


#
# produce output
# -------------------------------
#  

with open(args.out_file,"w",encoding="utf8") as out_file_text ,\
     open(args.in_file_ids,"r",encoding="utf8") as in_file_ids:

    for line in tqdm(in_file_ids):
        topicid, positive_docid, unjudged_docid = line.split()

        if collection[positive_docid].strip() != "" and collection[unjudged_docid].strip() != "":
            out_file_text.write(queries[topicid]+"\t"+collection[positive_docid]+"\t"+collection[unjudged_docid]+"\n")