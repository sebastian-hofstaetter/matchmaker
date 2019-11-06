#
# msmarco doc: create the train.tsv triples  
# -------------------------------
# try to mimic begavior of original https://github.com/microsoft/TREC-2019-Deep-Learning/blob/master/utils/msmarco-doctriples.py

import random
random.seed(42)

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *
from collections import defaultdict

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='training output text file location', required=True)

parser.add_argument('--out-file-ids', action='store', dest='out_file_ids',
                    help='training output ids file location', required=True)

parser.add_argument('--candidate-file', action='store', dest='candidate_file',
                    help='trec ranking file location (lucene output)', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

parser.add_argument('--qrel', action='store', dest='qrel_file',
                    help='qrel location', required=True)


args = parser.parse_args()


#
# load data 
# -------------------------------
# 
collection = {} # int id -> full line dictionary
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        collection[_id] = ls[1].rstrip()

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        queries[_id] = ls[1].rstrip()

qrels = load_reference(args.qrel_file)


#
# produce output
# -------------------------------
#  

unjudged_rank_to_keep = random.randint(1, 100)
already_done_a_triple_for_topicid = -1
stats = defaultdict(int)

with open(args.candidate_file,"r",encoding="utf8") as candidate_file, \
     open(args.out_file,"w",encoding="utf8") as out_file_text ,\
     open(args.out_file_ids,"w",encoding="utf8") as out_file_ids:

    for line in tqdm(candidate_file):
        [topicid, _, unjudged_docid, rank, _, _] = line.split()

        topicid = int(topicid)

        if already_done_a_triple_for_topicid == topicid or int(rank) != unjudged_rank_to_keep:
            stats['skipped'] += 1
            continue
        elif topicid not in queries: # added: because we carved out the validation qrels from the train -> so there are some missing
            stats['skipped'] += 1
            continue
        else:
            unjudged_rank_to_keep = random.randint(1, 100)
            already_done_a_triple_for_topicid = topicid

        assert topicid in qrels
        assert unjudged_docid in collection

        # Use topicid to get our positive_docid
        positive_docid = random.choice(qrels[topicid])
        assert positive_docid in collection

        if unjudged_docid in qrels[topicid]:
            stats['docid_collision'] += 1
            continue

        stats['kept'] += 1

        out_file_ids.write(str(topicid)+"\t"+positive_docid+"\t"+unjudged_docid+"\n")
        out_file_text.write(queries[topicid]+"\t"+collection[positive_docid]+"\t"+collection[unjudged_docid]+"\n")

for key, val in stats.items():
    print(f"{key}\t{val}")