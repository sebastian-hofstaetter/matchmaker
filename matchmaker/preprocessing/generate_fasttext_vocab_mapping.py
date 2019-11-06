#
# generate a vocab to subword id mapping for an allennlp vocab and fasttext model
# -> only valid for the given vocab + the given fasttext model
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import  Tqdm
Tqdm.default_mininterval = 1

import fastText
import statistics
#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--allen-vocab', action='store', dest='allen_vocab',
                    help='the tokens.txt file of the needed vocab', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='path to out file', required=True)

parser.add_argument('--fasttext-model', action='store', dest='model',
                    help='.bin model of fasttext', required=True)

args = parser.parse_args()


#
# load data & work
# -------------------------------
# 
model = fastText.load_model(args.model)

ind_stats = []

with open(args.allen_vocab,"r",encoding="utf8") as in_file:
    with open(args.out_file,"w",encoding="utf8") as out_file:
        for line in tqdm(in_file):
            line = line.strip()

            _, indices = model.get_subwords(line)
            ind_stats.append(len(indices))

            out_file.write(line + " " + " ".join(map(str, map(lambda x: x+1, indices)))+"\n")


print("avg.:",statistics.mean(ind_stats))
print("median.:",statistics.median(ind_stats))
print("min.:",min(ind_stats))
print("max.:",max(ind_stats))