from typing import Dict
from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
import copy
from matchmaker.dataloaders.transformer_tokenizer import *


# @DatasetReader.register("ConcatenatedReRankingDatasetReader")
class ConcatenatedReRankingDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create instances, where query and document are concatenated
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    
    The output of ``read`` is a list of ``Instance`` s with the fields:
        doc_tokens: ``TextField``

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. 
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """

    def __init__(self,
                 tokenizer: FastTransformerTokenizer,
                 token_indexers: Dict[str, TokenIndexer],

                 max_doc_length: int = -1,
                 max_query_length: int = -1,
                 min_doc_length: int = -1,
                 min_query_length: int = -1,
                 train_qa_spans: bool = False):

        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.min_doc_length = min_doc_length
        self.min_query_length = min_query_length

        self.train_qa_spans = train_qa_spans

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:

            for line in self.shard_iterable(data_file):
                line = line.strip("\n")
                line_parts = line.split('\t')

                qa_spans = None

                if self.train_qa_spans:
                    if len(line_parts) == 5:
                        query_id, doc_id, query_sequence, doc_sequence, qa_spans = line_parts
                    elif len(line_parts) == 4:
                        query_id, doc_id, query_sequence, doc_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line))
                else:
                    if len(line_parts) != 4:
                        raise ConfigurationError("Invalid line format: %s " % (line))
                    query_id, doc_id, query_sequence, doc_sequence = line_parts

                yield self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence, qa_spans)

    @overrides
    def text_to_instance(self, query_id: str, doc_id: str, query_sequence: str, doc_sequence: str, qa_spans: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        query_id_field = MetadataField(query_id)
        doc_id_field = MetadataField(doc_id)

        ret_instance = {
            "query_id": query_id_field,
            "doc_id": doc_id_field,
            "doc_tokens": PatchedTransformerTextField(**self._tokenizer.tokenize(query_sequence,doc_sequence,self.max_query_length + self.max_doc_length))}

        if self.train_qa_spans:
            pos_qa_start = []
            pos_qa_end = []

            pos_qa_hasAnswer = 0
            if qa_spans != None and qa_spans != "":
                pos_qa_hasAnswer = 1

                spans = qa_spans.split()
                q_offset = len(query_tokenized)
                for span in spans:
                    span = span.split(",")
                    span_start = int(span[0])
                    span_end = int(span[1])
                    got_start = False
                    got_end = False
                    for i, t in enumerate(doc_tokenized):
                        if t.idx_end == None:  # CLS tokens might return None
                            continue
                        if got_start == False and t.idx_end >= span_start:
                            # we've got the start!
                            pos_qa_start.append(q_offset + i)
                            got_start = True
                        if t.idx_end >= span_end:
                            # we've got the end!
                            pos_qa_end.append(q_offset + i)
                            got_end = True
                            break
                    if got_start and not got_end:        # we might have cut the ending of a span
                        pos_qa_end.append(q_offset + i)

                if len(pos_qa_start) != len(pos_qa_end):  # or len(pos_qa_start) != len(spans)
                    raise RuntimeError("error in qa span splitting: %s,%s" % (str(doc_tokenized), qa_spans))

            ret_instance["qa_start"] = ArrayField(np.array(pos_qa_start), padding_value=-1, dtype=np.int64)
            ret_instance["qa_end"] = ArrayField(np.array(pos_qa_end), padding_value=-1, dtype=np.int64)
            ret_instance["qa_hasAnswer"] = ArrayField(np.array(pos_qa_hasAnswer))

            ret_instance["offsets_start"] = MetadataField(offsets_start)
            ret_instance["offsets_end"] = MetadataField(offsets_end)
            ret_instance["query_text"] = MetadataField(query_sequence)
            ret_instance["doc_text"] = MetadataField(doc_sequence)

        return Instance(ret_instance)
