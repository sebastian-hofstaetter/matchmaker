#
# generate an idf file for re-use - of the full vocabulary & fasttext mapping (outputs matrix file, to be loaded with fasttextembeddingbag)
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import math
import numpy
from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from matchmaker.dataloaders.fasttext_token_indexer import *
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='idf nunmpy output file', required=True)

parser.add_argument('--fasttext-mapping', action='store', dest='fasttext_vocab',
                    help='fasttext vocab', required=True)

parser.add_argument('--fasttext-matrix-size', action='store', dest='fasttext_size',type=int,
                    help='fasttext total matrix size (including padding)', required=True) # 2950302 for the wiki-trigram 200 dim

parser.add_argument('--lowercase', action='store', dest='lowercase',type=bool,default=True,
                    help='bool', required=False)

parser.add_argument('--dataset-files', nargs='+', action='store', dest='dataset_files',
                    help='file format <id>\t<sequence text>', required=True)

args = parser.parse_args()


#
# load data & create idfs
# -------------------------------
#  
a,b=FastTextVocab.load_ids(args.fasttext_vocab,max_subwords=40)
fasttext_vocab = FastTextVocab(a,b,max_subwords=40)

loader = IrTupleDatasetReader(lazy=True,source_tokenizer=BlingFireTokenizer(),target_tokenizer=BlingFireTokenizer(),lowercase=args.lowercase)

total_documents=0
all_tokens={}

idf = numpy.ones((args.fasttext_size, 1), dtype=numpy.float32)

for file in args.dataset_files:
    for instance in Tqdm.tqdm(loader.read(file)):

        token_set = set([tok.text.lower() for tok in instance["target_tokens"].tokens])
        for token_text in token_set:

            mappings = fasttext_vocab.get_subword_ids(token_text)

            for map_id in mappings:
                idf[map_id] += 1

        total_documents += 1

idf = numpy.log(total_documents / idf)
idf[0] = 1.0 # clear padding value (so that it does not introduce artifacts for sloppy padding)
numpy.save(args.out_file,idf)
