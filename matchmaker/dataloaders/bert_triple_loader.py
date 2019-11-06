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


@DatasetReader.register("bert_triple_loader")
class BertTripleDatasetReader(DatasetReader):
    """

    BERT_cls-ready: concats the query and doc sequences with [SEP] in the middle

    Read a tsv file containing triple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
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
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.min_doc_length = min_doc_length
        self.min_query_length = min_query_length

        self.padding_value = Token(text = "[PAD]",text_id=0)
        self.sep_value = Token(text = "[SEP]")

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
                yield self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence)

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        query_tokenized = self._tokenizer.tokenize(query_sequence)
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]
        if self.min_query_length > -1 and len(query_tokenized) < self.min_query_length:
            query_tokenized = query_tokenized + [self.padding_value] * (self.min_query_length - len(query_tokenized))

        
        doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence)
        if self.max_doc_length > -1:
            doc_pos_tokenized = doc_pos_tokenized[:self.max_doc_length]
        if self.min_doc_length > -1 and len(doc_pos_tokenized) < self.min_doc_length:
            doc_pos_tokenized = doc_pos_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_pos_tokenized))

        doc_pos_field = TextField(query_tokenized + [self.sep_value] + doc_pos_tokenized, self._token_indexers)

        doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence)
        if self.max_doc_length > -1:
            doc_neg_tokenized = doc_neg_tokenized[:self.max_doc_length]
        if self.min_doc_length > -1 and len(doc_neg_tokenized) < self.min_doc_length:
            doc_neg_tokenized = doc_neg_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_neg_tokenized))

        doc_neg_field = TextField(query_tokenized + [self.sep_value] + doc_neg_tokenized, self._token_indexers)

        return Instance({
            "doc_pos_tokens":doc_pos_field,
            "doc_neg_tokens": doc_neg_field})