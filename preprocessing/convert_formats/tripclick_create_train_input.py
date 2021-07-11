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

max_triples = 10_000_000
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

candidates = defaultdict(list)

with open(args.candidate_file,"r",encoding="utf8") as candidate_file:

    for line in tqdm(candidate_file):
        #if random.random() <= 0.5: continue #skip some entries for faster processing
        [topicid, _ , unjudged_docid, rank, _ , _ ] = line.split()
        
        candidates[topicid].append(unjudged_docid)

for q, docs in tqdm(qrels.items()):

    d_set = set(docs)
    for d in docs:
        if q not in candidates:
            stats['missing candidates'] += 1

        cand = candidates[q]

        negative_samples = random.sample(cand, min(20,len(cand)))

        for n in negative_samples:

            if n in d_set:
                stats['docid_collision'] += 1
                continue

            stats['kept'] += 1

            triples.append((q,d,n))

        #if int(rank) <= 100:
        #    #if random.random() < 0.7: continue # skip 70% of candidates to speed up things...
        #    #else:
        #    stats['< 100 sampling count'] += 1
        #else:
        #    if random.random() <= 0.9: continue # skip 90% of candidates assumong top1k -> same number of samples from 0-100 as 101 - 1000
        #    else:
        #        stats['> 100 sampling count'] += 1

        #if topicid not in queries or topicid not in qrels: # added: because we carved out the validation qrels from the train -> so there are some missing
        #    stats['skipped'] += 1
        #    continue
#
        ##assert topicid in qrels
        #assert unjudged_docid in collection
#
        ## Use topicid to get our positive_docid
        #positive_docid = random.choice(qrels[topicid])
        #assert positive_docid in collection
#
        #if unjudged_docid in qrels[topicid]:
        #    stats['docid_collision'] += 1
        #    continue
#
        #stats['kept'] += 1
#
        ##if collection_length[positive_docid] > max_doc_token_length and collection_length[unjudged_docid] > max_doc_token_length:
        ##    stats['both_to_long'] += 1
        ##    continue
        ##if collection_length[positive_docid] > max_doc_token_length:
        ##    stats['pos_to_long'] += 1
        ##    continue
        ##if collection_length[unjudged_docid] > max_doc_token_length:
        ##    stats['unjuged_to_long'] += 1
        ##    continue
#
        #triples.append((topicid,positive_docid,unjudged_docid))

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