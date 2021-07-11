#
# trec car -> topic to msmarco queries 
#
# topics example:  
# enwiki:Antibiotics/Medical%20uses/Administration
# enwiki:Antibiotics/Side-effects
# enwiki:Antibiotics/Side-effects/Obesity
# 
# (can also be used with trec car qrels -> to generate queries that are judged)
# -------------------------------
#

import urllib.parse
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())


#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='trec car topic file format', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='passage query format', required=True)

args = parser.parse_args()


#
# work 
#
known_ids={}

with open(args.in_file,"r",encoding="utf8") as in_file:
    with open(args.out_file,"w",encoding="utf8") as out_file:
        for line in tqdm(in_file):
            
            if " " in line:
                qid = line.strip().split()[0]
            else:
                qid = line.strip()

            if qid not in known_ids:
                known_ids[qid] = 0
                query = urllib.parse.unquote(qid).replace("enwiki:","").replace("/"," ")
                out_file.write(qid+"\t"+query.replace("\t"," ").replace("\n"," ").strip()+"\n")