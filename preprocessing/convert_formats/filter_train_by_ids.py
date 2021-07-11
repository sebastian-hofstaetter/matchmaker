import argparse
from tqdm import tqdm
import os

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='output folder', required=True )

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output folder', required=True)

parser.add_argument('--query-ids', action='store', dest='q_ids',
                    help='output folder', required=True)

parser.add_argument('--query-text', action='store', dest='q_text',
                    help='output folder', required=True)

args = parser.parse_args()


#
# work 
#

query_ids = set()
with open(args.q_ids,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        query_ids.add(line.strip())

queries = set()
with open(args.q_text,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        if _id in query_ids:
            queries.add(ls[1].rstrip())
count=0
with open(args.in_file,"r",encoding="utf8") as in_file,\
     open(args.out_file,"w",encoding="utf8") as out_file:
    for line in tqdm(in_file):
        if line.split("\t")[0] in queries:
            out_file.write(line)
            count+=1

print("count remaining",count)