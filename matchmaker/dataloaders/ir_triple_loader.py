# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict
import logging

from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,LabelField,ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from blingfire import *

import random
import math

@DatasetReader.register("ir_triple_loader")
class IrTripleDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 min_doc_length:int = -1,
                 min_query_length:int = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.min_doc_length = min_doc_length
        self.min_query_length = min_query_length

        self.padding_value = Token(text = "@@PADDING@@",text_id=0)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                inst =  self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence)
                if inst is not None:
                    yield inst

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        #doc_pos_sequence = text_to_sentences(doc_pos_sequence).split("\n")
        #doc_neg_sequence = text_to_sentences(doc_neg_sequence).split("\n")
#
        #random.shuffle(doc_pos_sequence)
        #random.shuffle(doc_neg_sequence)
#
        #doc_pos_sequence = " ".join(doc_pos_sequence)
        #doc_neg_sequence = " ".join(doc_neg_sequence)

        query_tokenized = self._tokenizer.tokenize(query_sequence)
        #if self._source_add_start_token:
        #    query_tokenized.insert(0, Token(START_SYMBOL))
        #query_tokenized.append(Token(END_SYMBOL))
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]
        if self.min_query_length > -1 and len(query_tokenized) < self.min_query_length:
            query_tokenized = query_tokenized + [self.padding_value] * (self.min_query_length - len(query_tokenized))

        query_field = TextField(query_tokenized, self._token_indexers)
        
        doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence)
        #doc_pos_tokenized.insert(0, Token(START_SYMBOL))
        #doc_pos_tokenized.append(Token(END_SYMBOL))
        if self.max_doc_length > -1:
            doc_pos_tokenized = doc_pos_tokenized[:self.max_doc_length]
        if self.min_doc_length > -1 and len(doc_pos_tokenized) < self.min_doc_length:
            doc_pos_tokenized = doc_pos_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_pos_tokenized))

        #if random.random() > 0.5:
            #half = int(len(doc_pos_tokenized))
            #part1 = doc_pos_tokenized[:half]
            #doc_pos_tokenized = doc_pos_tokenized[half:] + part1
        #parts = int(math.ceil(len(doc_pos_tokenized) / 20))
        #doc_pos_tokenized_new = []
        #part_order = list(range(parts))
        #random.shuffle(part_order)
        #for i in part_order:
        #    doc_pos_tokenized_new += doc_pos_tokenized[i*20:(i+1)*20]
        #print(doc_pos_tokenized,"stop", doc_pos_tokenized_new)
        #print(doc_pos_tokenized_new)
        #doc_pos_tokenized = doc_pos_tokenized_new
        

        doc_pos_field = TextField(doc_pos_tokenized, self._token_indexers)

        doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence)
        #doc_neg_tokenized.insert(0, Token(START_SYMBOL))
        #doc_neg_tokenized.append(Token(END_SYMBOL))
        if self.max_doc_length > -1:
            doc_neg_tokenized = doc_neg_tokenized[:self.max_doc_length]
        if self.min_doc_length > -1 and len(doc_neg_tokenized) < self.min_doc_length:
            doc_neg_tokenized = doc_neg_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_neg_tokenized))

        #if random.random() > 0.5:
        #    half = int(len(doc_neg_tokenized))
        #    part1 = doc_neg_tokenized[:half]
        #    doc_neg_tokenized = doc_neg_tokenized[half:] + part1
        #parts = int(math.ceil(len(doc_neg_tokenized) / 20.0))
        #doc_neg_tokenized_new = []
        #part_order = list(range(parts))
        #random.shuffle(part_order)
        #for i in part_order:
        #    doc_neg_tokenized_new += doc_neg_tokenized[i*20:(i+1)*20]
        #print(doc_pos_tokenized,"stop", doc_pos_tokenized_new)
        #print(doc_pos_tokenized_new)
        #doc_neg_tokenized = doc_neg_tokenized_new
        #doc_neg_tokenized.reverse()

        doc_neg_field = TextField(doc_neg_tokenized, self._token_indexers)

        if len(query_tokenized) == 0 or len(doc_neg_tokenized) == 0 or len(doc_pos_tokenized) == 0:
            return None

        return Instance({
            "query_tokens":query_field,
            "doc_pos_tokens":doc_pos_field,
            "doc_neg_tokens": doc_neg_field})