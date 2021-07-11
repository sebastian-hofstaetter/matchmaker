# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict,Union
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,MetadataField,TransformerTextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from matchmaker.dataloaders.transformer_tokenizer import *


class IdSequenceDatasetReader(DatasetReader):
    """
    Read a tsv file containing a single sequence <sequence_id>\t<sequence_string>
    """
    def __init__(self,
                 tokenizer: Union[Tokenizer,FastTransformerTokenizer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_length:int = -1,
                 min_seq_length:int = -1,
                 sequence_type:str = "doc") -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.sequence_type = sequence_type
        #self.make_multiple_of = 8

        if isinstance(tokenizer, FastTransformerTokenizer):
            self.token_type = "huggingface"
        else:
            self.token_type = "emb"
            self.padding_value = Token(text = "@@PADDING@@",text_id=0)
            self.cls_value = Token(text = "@@UNKNOWN@@",text_id=1)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            for line in self.shard_iterable(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format, you have too many or too little columns in the line - maybe clean excess tabs from the text? Line: %s " % (line))

                seq_id, seq_text = line_parts
                yield self.text_to_instance(seq_id, seq_text)

    @overrides
    def text_to_instance(self, seq_id:str, seq_text:str) -> Instance:

        seq_id_field = MetadataField(seq_id)
        
        if self.token_type == "huggingface":
            seq_tokenized = self._tokenizer.tokenize(seq_text, max_length=self.max_seq_length)
            seq_field = PatchedTransformerTextField(**seq_tokenized)

        else:
            seq_tokenized = self._tokenizer.tokenize(seq_text)

            if self.max_seq_length > -1:
                seq_tokenized = seq_tokenized[:self.max_seq_length]
            if self.min_seq_length > -1 and len(seq_tokenized) < self.min_seq_length:
                seq_tokenized = seq_tokenized + [self.padding_value] * (self.min_seq_length - len(seq_tokenized))

            #if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            #    seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)

            seq_field = TextField(seq_tokenized)
        
        return Instance({
            "seq_id":seq_id_field,
            "seq_tokens":seq_field})

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        if self.token_type != "huggingface":
            instance.fields["seq_tokens"]._token_indexers = self._token_indexers  # type: ignore