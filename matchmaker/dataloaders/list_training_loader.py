from collections import defaultdict
from typing import Dict
import logging
import numpy

from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,LabelField,ArrayField,MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from allennlp.data.samplers import BatchSampler
from matchmaker.utils.core_metrics import *
from allennlp.data.batch import Batch
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

from blingfire import *

import random
import math
from tqdm import tqdm


class IrDynamicTripleDatasetLoader():
    """
    dynamically samples 1 pos (from judged qrels) and batch_size neg (from candidate file) documents per query
    """

    def __init__(
        self,
        query_file:str,
        collection_file,
        qrels_file,
        candidate_file,
        batch_size: int,
        queries_per_batch,
        vocab,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        max_doc_length:int = -1,
        max_query_length:int = -1,
        min_doc_length:int = -1,
        min_query_length:int = -1,
        data_augment:str="none",
        make_multiple_of:int=-1,
    ):
        self.candidate_file=candidate_file
        self.batch_size = batch_size
        self.queries_per_batch = queries_per_batch
        
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.min_doc_length = min_doc_length
        self.min_query_length = min_query_length
        self.data_augment = data_augment
        self.vocab = vocab
        #if type(tokenizer) == CustomTransformerTokenizer:
        #    self.token_type = "bpe"
        #    self.padding_value = Token(text = "<pad>", text_id=tokenizer.tokenizer.token_to_id("<pad>"))
        #    self.cls_value = Token(text = "<s>", text_id=tokenizer.tokenizer.token_to_id("<s>")) # forgot to add <cls> to the vocabs
        #else:
        #    self.padding_value = Token(text = "@@PADDING@@",text_id=0)
        #    self.cls_value = Token(text = "@@UNKNOWN@@",text_id=1)

        self.lucene_stopwords=set(["a", "an", "and", "are", "as", "at", "be", "but", "by","for", "if", "in", "into", "is", "it","no", "not", "of", "on", "or", "such","that", "the", "their", "then", "there", "these","they", "this", "to", "was", "will", "with"])
        self.msmarco_top_50_stopwords=set([".","the",",","of","and","a","to","in","is","-","for",")","(","or","you","that","are","on","it",":","as","with","your","'s","from","by","be","can","an","this","1","at","have","2","not","/","was","if","$","will","i","one","which","more","has","3","but","when","what","all"])
        self.msmarco_top_25_stopwords=set([".","the",",","of","and","a","to","in","is","-","for",")","(","or","you","that","are","on","it",":","as","with","your","'s","from"])
        self.use_stopwords = False
        self.make_multiple_of = make_multiple_of
        
        self.max_title_length = 30
        self.min_title_length = -1
        self.add_cls_to_query = True

        #
        # load data - init
        #
        self.candidates = load_ranking(candidate_file)
        self.qrels = load_qrels(qrels_file)
        self.query_ids = list(set(self.qrels.keys()).intersection(set(self.candidates.keys())))
        print("Got ",len(self.query_ids)," queries to sample for rnd_state:",random.randint(0,100))

        self.collection = {}
        self.collection_ids = []
        with open(collection_file,"r",encoding="utf8") as cf:
            for line in cf:
                ls = line.split("\t") # id<\t>text ....
                self.collection[ls[0]] = ls[1].rstrip()[:100_000]
                self.collection_ids.append(ls[0])

        self.queries = {}
        with open(query_file,"r",encoding="utf8") as qf:
            for line in qf:
                ls = line.split("\t") # id<\t>text ....
                self.queries[ls[0]] = ls[1].rstrip()
        
        self.empty_cls_field = TextField([self.cls_value], self._token_indexers)
        self.empty_cls_field.index(self.vocab)

    def __iter__(self):
        #batch_count=0
        #next_query_update=2_000
        while True:
            main_instances = []
            candidate_ids = []
            candidate_texts = []
            for i in range(self.queries_per_batch):

                candidate_target_count = int((self.batch_size / self.queries_per_batch) // 2)
                random_target_count = int((self.batch_size // self.queries_per_batch)) - candidate_target_count

                q_id = random.choice(self.query_ids)
                query_text = self.get_tokenized_query(self.queries[q_id])
                query_text.index(self.vocab)

                all_pos_ids = list(self.qrels[q_id].keys())
                pos_id = random.choice(all_pos_ids)
                pos_text = self.get_tokenized_document(self.collection[pos_id])
                pos_text.index(self.vocab)

                for pi in all_pos_ids:
                    if pi in self.candidates[q_id]: 
                        self.candidates[q_id].remove(pi)

                # if we don't have enough candidates, skip
                if len(self.candidates[q_id]) <= candidate_target_count:
                    continue
    

                main_instances.append(Instance({
                "query_tokens":query_text,
                "title_pos_tokens":self.empty_cls_field,
                "doc_pos_tokens":pos_text,
                # first is pos, then candidate samples, then random samples
                "labels":ArrayField(np.array([3] + [1]*candidate_target_count + [0]*random_target_count))}))

                for cand_id in random.sample(self.candidates[q_id], candidate_target_count) :
                    candidate_ids.append(cand_id)
                    cand_text=self.get_tokenized_document(self.collection[cand_id])
                    cand_text.index(self.vocab)
                    candidate_texts.append(Instance({"title_neg_tokens":self.empty_cls_field,
                                                     "doc_neg_tokens": cand_text}))

                for cand_id in random.sample(self.collection_ids, random_target_count) :
                    candidate_ids.append(cand_id)
                    cand_text=self.get_tokenized_document(self.collection[cand_id])
                    cand_text.index(self.vocab)
                    candidate_texts.append(Instance({"title_neg_tokens":self.empty_cls_field,
                                                     "doc_neg_tokens": cand_text}))
            #batch_count += 1
            #if batch_count == next_query_update:
            #    self.queries_per_batch //= 2
            #    if self.queries_per_batch < 2: self.queries_per_batch = 1
            #    next_query_update *= 2
            #    print("Now using queries_per_batch: ",self.queries_per_batch)

            neg_batch = Batch(candidate_texts)
            neg_batch = neg_batch.as_tensor_dict(neg_batch.get_padding_lengths())

            main_batch = Batch(main_instances)
            main_batch = main_batch.as_tensor_dict(main_batch.get_padding_lengths())

            main_batch.update(neg_batch)
            yield main_batch

    def get_tokenized_query(self, text):
        query_tokenized = self._tokenizer.tokenize(text)

        if self.add_cls_to_query:
            query_tokenized.insert(0,self.cls_value)

        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]
        if self.min_query_length > -1 and len(query_tokenized) < self.min_query_length:
            query_tokenized = query_tokenized + [self.padding_value] * (self.min_query_length - len(query_tokenized))

        if self.make_multiple_of > -1 and len(query_tokenized) % self.make_multiple_of != 0:
            query_tokenized = query_tokenized + [self.padding_value] * (self.make_multiple_of - len(query_tokenized) % self.make_multiple_of)

        return TextField(query_tokenized, self._token_indexers)

    def get_tokenized_document(self, text):
        doc_tokenized = self._tokenizer.tokenize(text)

        if self.max_doc_length > -1:
            doc_tokenized = doc_tokenized[:self.max_doc_length]
        if self.min_doc_length > -1 and len(doc_tokenized) < self.min_doc_length:
            doc_tokenized = doc_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_tokenized))

        if self.make_multiple_of > -1 and len(doc_tokenized) % self.make_multiple_of != 0:
            doc_tokenized = doc_tokenized + [self.padding_value] * (self.make_multiple_of - len(doc_tokenized) % self.make_multiple_of)

        return TextField(doc_tokenized, self._token_indexers)
