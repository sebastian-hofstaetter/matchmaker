#
# msmarco doc collection format to passage (fewer columns)
# 
# * doc collection.tsv: docid, url, title, body
# * passage collection.tsv: docid, concat(title,body)
#  
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='doc collection', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='passage collection format', required=True)

args = parser.parse_args()


#
# work 
#

with open(args.in_file,"r",encoding="utf8") as in_file:
    with open(args.out_file,"w",encoding="utf8") as out_file:
        for line in tqdm(in_file):

            ls = line.split("\t") # docid, url, title, body
            out_file.write(ls[0]+"\t"+ls[2]+" "+ls[3]) # \n is part of ls[3]