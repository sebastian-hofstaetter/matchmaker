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


#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='val_out_file',
                    help='validation output file location', required=True)

parser.add_argument('--candidate-files', nargs='+', action='store', dest='candidate_file',
                    help='trec ranking file location (lucene output)', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

args = parser.parse_args()

max_doc_char_length = 100_000


#
# load data 
# -------------------------------
# 
collection = {} # int id -> full line dictionary
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        collection[_id] = ls[1].rstrip()[:max_doc_char_length]# +"\t"+ls[2].rstrip()[:max_doc_char_length]

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
known_pairs = set()

with open(args.val_out_file,"w",encoding="utf8") as val_out_file:
    for f in tqdm(args.candidate_file):
        with open(f,"r",encoding="utf8") as candidate_file:

            for line in tqdm(candidate_file):
                ls = line.split()
                if len(ls) == 4:
                    query_id = ls[0]
                    doc_id = ls[1]
                else:
                    query_id = ls[0]
                    doc_id = ls[2]
  
                if (query_id,doc_id) not in known_pairs:
                    known_pairs.add((query_id,doc_id))

                    out_arr = [query_id,doc_id,queries[query_id],collection[doc_id]]

                    val_out_file.write("\t".join(out_arr)+"\n")