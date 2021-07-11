#
# Take .tsv collection files with fulltext and score-id files to produce score-text files
# -------------------------------
#

import argparse
from collections import defaultdict
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--collection', action='store', dest='collection',
                    help='The full collection file: <did text>', required=True)

parser.add_argument('--query', action='store', dest='query',
                    help='The query text file: <qid text>', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='The teacher training score file: <s_pos s_neg q_id d_pos_id d_neg_id>', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='The teacher training score file output (filled with text): <s_pos s_neg q_text d_pos_text d_neg_text>', required=True)


args = parser.parse_args()


#
# load data 
# -------------------------------
# 
queries = {}
with open(args.query,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        queries[ls[0]] = ls[1].rstrip()

docs = {}
with open(args.collection,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        docs[ls[0]] = ls[1].rstrip()

#
# produce output
# -------------------------------
#  
stats = defaultdict(int)
with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.in_file,"r",encoding="utf8") as in_file:

        for line in tqdm(in_file):
            line = line.split("\t") # scorpos scoreneg query docpos docneg

            try:
                q_text = queries[line[2]]
                doc_pos_text = docs[line[3]]
                doc_neg_text = docs[line[4].rstrip()]
    
                out_file.write(line[0]+"\t"+line[1]+"\t"+q_text+"\t"+doc_pos_text+"\t"+doc_neg_text+"\n")

                stats["success"]+=1
            except KeyError as e:
                stats["key_error"]+=1

for key, val in stats.items():
    print(f"{key}\t{val}")