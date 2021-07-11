#
# generate ensembled pairwise teacher scores from a number of train-id files
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
                    help='the full collection file location', required=True)

parser.add_argument('--query', action='store', dest='query',
                    help='query.tsv', required=True)

parser.add_argument('--in-files', action='store', dest='in_files',
                    help='the teacher score - id files', required=True)

parser.add_argument('--out-file-ordering', action='store', dest='out_file_ordering',
                    help='teacher score out file (only taking the ordering)', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='triple text + ensemble score', required=True)


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



data_per_qid = {}

all_pairs = set()

for dp in args.in_files.split(","):
    name = dp
    with open(dp,"r",encoding="utf8") as in_file:
        for i,line in tqdm(enumerate(in_file)):
            line = line.split("\t") 

            pair = (line[2],line[3])
            if pair not in data_per_qid:
                data_per_qid[pair]={}
            data_per_qid[pair][name]=float(line[0])

            pair2 = (line[2],line[4].rstrip())
            if pair2 not in data_per_qid:
                data_per_qid[pair2]={}
            data_per_qid[pair2][name]=float(line[1])

            all_pairs.add((pair,pair2))

ensemble_type = "mean"

if ensemble_type == "mean":

    for key, val in data_per_qid.items():
        mean = sum(val.values()) / len(val.values())
        data_per_qid[key] = mean


#
# produce output
# -------------------------------
#  
stats = defaultdict(int)
with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.out_file_ordering,"r",encoding="utf8") as out_file_ordering:

        for line in tqdm(out_file_ordering):
            line = line.split("\t") # scorpos scoreneg query docpos docneg

            pos_pair = (line[2],line[3])
            neg_pair = (line[2],line[4].rstrip())

            try:

                out_file.write(str(data_per_qid[pos_pair])+"\t"+str(data_per_qid[neg_pair])+"\t"+queries[line[2]]+"\t"+docs[line[3]]+"\t"+docs[line[4].rstrip()]+"\n")

            except KeyError as e:
                stats["key_error"]+=1


for key, val in stats.items():
    print(f"{key}\t{val}")