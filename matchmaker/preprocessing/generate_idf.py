#
# generate an idf file for re-use - of the full vocabulary
# -------------------------------
#

# usage:

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import math
from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_dir',
                    help='idf "embedding-like" output file', required=True)

parser.add_argument('--lowercase', action='store', dest='lowercase',type=bool,default=True,
                    help='bool', required=False)

parser.add_argument('--dataset-files', nargs='+', action='store', dest='dataset_files',
                    help='file format <id>\t<sequence text>', required=True)

args = parser.parse_args()


#
# load data & create vocab
# -------------------------------
#  

loader = IrTupleDatasetReader(lazy=True,source_tokenizer=BlingFireTokenizer(),target_tokenizer=BlingFireTokenizer(),lowercase=args.lowercase)

total_documents=0
all_tokens={}

for file in args.dataset_files:
    for instance in Tqdm.tqdm(loader.read(file)):

        token_set = set([tok.text.lower() for tok in instance["target_tokens"].tokens])
        for token_text in token_set:
            if token_text not in all_tokens:
                all_tokens[token_text]=0
            all_tokens[token_text]+=1

        total_documents += 1

with open(args.out_dir,"w",encoding="utf8") as out:
    for token,count in all_tokens.items():
        out.write(token+" "+f'{math.log(total_documents/count):1.20f}'+"\n")