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

max_triples = 5_000_000
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
        collection[_id] = max_char_doc
        #collection_length[_id] = len(tokenizer.tokenize(max_char_doc))

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        queries[_id] = ls[1].rstrip()

qrels = load_reference(args.qrel_file)


#
# produce output
# -------------------------------
#  

triples = []

stats = defaultdict(int)

bm25_candidates = {}
with open(args.candidate_file,"r",encoding="utf8") as candidate_file:

    for line in tqdm(candidate_file):
        [topicid, _, unjudged_docid, rank, _, _] = line.split()
        if topicid in qrels:
            if topicid not in bm25_candidates:
                bm25_candidates[topicid]=[]

            if unjudged_docid not in qrels[topicid]:
                bm25_candidates[topicid].append(unjudged_docid)

print("len(bm25_candidates)",len(bm25_candidates))

uniform_draw_per_query = 10
draw_from_bm25_per_query = 0

collection_ids = list(collection.keys())

for topicid in tqdm(qrels):

    if topicid not in queries: # added: because we carved out the validation qrels from the train -> so there are some missing
        stats['skipped'] += 1
        continue

    for i in range(uniform_draw_per_query):
        positive_docid = random.choice(qrels[topicid])
        neg_docid = random.choice(collection_ids)
        
        if neg_docid in qrels[topicid]:
            stats['docid_collision'] += 1
            continue

        stats['kept'] += 1

        triples.append((topicid,positive_docid,neg_docid))

    if topicid not in bm25_candidates:
        stats['not_in_bm25'] += 1
        continue

    for i in range(draw_from_bm25_per_query):
        positive_docid = random.choice(qrels[topicid])
        neg_docid = random.choice(bm25_candidates[topicid])
        
        if neg_docid in qrels[topicid]:
            stats['docid_collision'] += 1
            continue

        stats['kept'] += 1

        triples.append((topicid,positive_docid,neg_docid))

    if len(triples) > max_triples:
        break

# important: shuffle the train data
random.shuffle(triples)


with open(args.out_file,"w",encoding="utf8") as out_file_text ,\
     open(args.out_file_ids,"w",encoding="utf8") as out_file_ids:

    for i,(topicid, positive_docid, unjudged_docid) in tqdm(enumerate(triples)):
        if i == max_triples:
            break
        if collection[positive_docid].strip() != "" and collection[unjudged_docid].strip() != "":
            out_file_ids.write(str(topicid)+"\t"+positive_docid+"\t"+unjudged_docid+"\n")
            out_file_text.write(queries[topicid]+"\t"+collection[positive_docid]+"\t"+collection[unjudged_docid]+"\n")

for key, val in stats.items():
    print(f"{key}\t{val}")