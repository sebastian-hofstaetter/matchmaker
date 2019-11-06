#
# benchmark setup to measer time per component in the batch generation 
# -------------------------------
#
# usage:

import argparse
import os
import sys
sys.path.append(os.getcwd())

from matchmaker.dataloaders.ir_triple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from matchmaker.utils import *
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from matchmaker.dataloaders.fasttext_token_indexer import FastTextVocab,FastTextNGramIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer


#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--dataset-file', action='store', dest='dataset_file',
                    help='dataset file: for triple loader', required=True)
parser.add_argument('--vocab-file', action='store', dest='vocab_file',
                    help='vocab directory path', required=True)

args = parser.parse_args()


#
# load data & create vocab
# -------------------------------
#  
#_token_indexers = {"tokens": FastTextNGramIndexer(20)}
#_token_indexers = {"tokens": FastTextNGramIndexer(20)}
#_token_indexers = {"tokens": ELMoTokenCharactersIndexer()}

loader = IrTripleDatasetReader(lazy=True,#token_indexers=_token_indexers,
tokenizer=BlingFireTokenizer()) #BlingFireTokenizer()) #WordTokenizer(word_splitter=JustSpacesWordSplitter()))
#,max_doc_length=200,max_query_length=20,min_doc_length=200,min_query_length=20)

instances = loader.read(args.dataset_file)
_iterator = BucketIterator(batch_size=64,
                           sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

#vocab_map,vocab_data = FastTextVocab.load_ids(args.vocab_file,20)

#vocab = FastTextVocab(vocab_map, vocab_data,20)

_iterator.index_with(Vocabulary.from_files(args.vocab_file))

with Timer("iterate over all"):
    for i in _iterator(instances, num_epochs=1):
        exit()