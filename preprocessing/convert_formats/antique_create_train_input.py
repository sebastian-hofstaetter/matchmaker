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

from matchmaker.core_metrics import load_qrels
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
max_doc_char_length = 70_000

max_doc_token_length = 800

#
# load data 
# -------------------------------
# 
collection = {}
collection_length = {} 
tokenizer = BlingFireTokenizer()
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        max_char_doc = ls[1].rstrip()[:max_doc_char_length]
        collection[_id] = max_char_doc
        collection_length[_id] = len(tokenizer.tokenize(max_char_doc))

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        queries[_id] = ls[1].rstrip()

qrels = load_qrels(args.qrel_file)


#
# produce output
# -------------------------------
#  

triples = []

stats = defaultdict(int)

with open(args.candidate_file,"r",encoding="utf8") as candidate_file:

    for line in tqdm(candidate_file):
        [topicid, _, unjudged_docid, rank, _, _] = line.split()

        if topicid not in queries: # added: because we carved out the validation qrels from the train -> so there are some missing
            stats['skipped'] += 1
            continue

        assert topicid in qrels
        assert unjudged_docid in collection

        positive_docid = random.choice(list(qrels[topicid].keys()))
        unjudged_score = 0
        if unjudged_docid in qrels[topicid]:
            unjudged_score = qrels[topicid][unjudged_docid]

        if len(qrels[topicid]) > 1:
            trys = 0
            # Use topicid to get our positive_docid
            while positive_docid == unjudged_docid and qrels[topicid][positive_docid] <= unjudged_score:
                positive_docid = random.choice(list(qrels[topicid].keys()))
                assert positive_docid in collection
                trys+=1
                if trys>5:
                    break

        if positive_docid == topicid:
            stats['docid_collision'] += 1
            continue

        if qrels[topicid][positive_docid] <= unjudged_score:
            stats['docid_collision'] += 1
            continue

        stats['kept'] += 1

        #if collection_length[positive_docid] > max_doc_token_length and collection_length[unjudged_docid] > max_doc_token_length:
        #    stats['both_to_long'] += 1
        #    continue
        #if collection_length[positive_docid] > max_doc_token_length:
        #    stats['pos_to_long'] += 1
        #    continue
        #if collection_length[unjudged_docid] > max_doc_token_length:
        #    stats['unjuged_to_long'] += 1
        #    continue

        triples.append((topicid,positive_docid,unjudged_docid))

# important: shuffle the train data
random.shuffle(triples)


with open(args.out_file,"w",encoding="utf8") as out_file_text ,\
     open(args.out_file_ids,"w",encoding="utf8") as out_file_ids:

    for i,(topicid, positive_docid, unjudged_docid) in tqdm(enumerate(triples)):
        if i == max_triples:
            break
        out_file_ids.write(str(topicid)+"\t"+positive_docid+"\t"+unjudged_docid+"\n")
        out_file_text.write(queries[topicid]+"\t"+collection[positive_docid]+"\t"+collection[unjudged_docid]+"\n")

for key, val in stats.items():
    print(f"{key}\t{val}")