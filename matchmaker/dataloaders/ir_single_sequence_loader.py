# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict
import logging

from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,MetadataField,ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ir_single_seq_loader")
class IrSingleSequenceDatasetReader(DatasetReader):
    """
    Read a tsv file containing a single sequence <sequence_id>\t<sequence_string>
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_length:int = -1,
                 min_seq_length:int = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length

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
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                seq_id, seq_text = line_parts
                yield self.text_to_instance(seq_id, seq_text)

    @overrides
    def text_to_instance(self, seq_id:str, seq_text:str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        seq_id_field = MetadataField(seq_id)

        seq_tokenized = self._tokenizer.tokenize(seq_text)

        if self.max_seq_length > -1:
            seq_tokenized = seq_tokenized[:self.max_seq_length]
        if self.min_seq_length > -1 and len(seq_tokenized) < self.min_seq_length:
            seq_tokenized = seq_tokenized + [self.padding_value] * (self.min_seq_length - len(seq_tokenized))

        seq_field = TextField(seq_tokenized, self._token_indexers)
        
        return Instance({
            "seq_id":seq_id_field,
            "seq_tokens":seq_field})
