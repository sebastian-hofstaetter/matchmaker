#
# create a pre-trained embedding with gensim w2v - using the output of tokenize_files.py (spacy tokenizer)
# -------------------------------

import argparse
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm

import gensim

from matchmaker.dataloaders.ir_labeled_tuple_loader import *
from matchmaker.dataloaders.ir_tuple_loader import *
from matchmaker.dataloaders.ir_triple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output file', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='input file', required=True)

args = parser.parse_args()


class MySentences(object):
    def __init__(self, file):
        self.file = file
 
    def __iter__(self):
        with open(self.file,"r",encoding="utf8") as file:
            for line in file:
                yield line.split()

model = gensim.models.Word2Vec(MySentences(args.in_file), min_count=10, size=300, iter=25, window=5)

model.wv.save_word2vec_format(args.out_file)