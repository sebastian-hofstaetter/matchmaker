from typing import Dict
from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,ArrayField,MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
import copy
import random
from blingfire import *
from matchmaker.dataloaders.transformer_tokenizer import *

#@DatasetReader.register("bert_triple_loader")
class ConcatenatedTrainingDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a
    a concatenated re-ranking model; concats the query and doc sequences with [SEP] in the middle.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    
    The output of ``read`` is a list of ``Instance`` s with the fields:
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. 
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations.
    """
    def __init__(self,
                 tokenizer: FastTransformerTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,

                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 min_doc_length:int = -1,
                 min_query_length:int = -1,

                 data_augment:str="none",

                 train_pairwise_distillation:bool = False,
                 train_qa_spans:bool = False):

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
        self.data_augment = data_augment

        self.add_text_to_batch = True
        self.read_with_scores = train_pairwise_distillation
        self.train_qa_spans = train_qa_spans
        self.qa_skip_noqa = False

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            for line in self.shard_iterable(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                pos_score= None
                neg_score = None
                qa_spans_pos = None

                if not self.read_with_scores and not self.train_qa_spans:
                    if len(line_parts) == 3:
                        query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line))
                elif self.train_qa_spans:
                    if len(line_parts) == 4:
                        qa_spans_pos,query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line))
                else:
                    if len(line_parts) == 5:
                        pos_score,neg_score, query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line)) 

                inst = self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence,pos_score,neg_score,qa_spans_pos)
                if inst is not None: # this should not happen (but just in case we don't break training)
                    yield inst

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str,pos_score,neg_score,qa_spans_pos) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        def data_augment(aug_type,string):

            if aug_type == "shuffle_sent":
                doc_sequence = text_to_sentences(string).split("\n")
                random.shuffle(doc_sequence)
                doc_sequence = " ".join(doc_sequence)

            elif aug_type == "reverse_sent":
                doc_sequence = text_to_sentences(string).split("\n")
                doc_sequence = " ".join(doc_sequence[::-1])

            elif aug_type == "rotate":
                tokens = text_to_words(string).split()
                n = random.randint(0,len(tokens)-1)
                doc_sequence = " ".join(tokens[n:] + tokens[:n])

            elif aug_type == "none":
                doc_sequence = string
            else:
                raise Exception("wrong aug_type")

            return doc_sequence

        if self.data_augment != "none":
            if len(doc_pos_sequence) == 0 or len(doc_neg_sequence) == 0:
                return None
            doc_pos_sequence = data_augment(self.data_augment,doc_pos_sequence)
            doc_neg_sequence = data_augment(self.data_augment,doc_neg_sequence)

        ret_instance = {
                "doc_pos_tokens": PatchedTransformerTextField(**self._tokenizer.tokenize(query_sequence,doc_pos_sequence,self.max_query_length + self.max_doc_length)),
                "doc_neg_tokens": PatchedTransformerTextField(**self._tokenizer.tokenize(query_sequence,doc_neg_sequence,self.max_query_length + self.max_doc_length))}

        if self.train_qa_spans:
            pos_qa_start = []
            pos_qa_end = []

            pos_qa_hasAnswer = 0
            if qa_spans_pos != None and qa_spans_pos != "":
                pos_qa_hasAnswer = 1

                spans = qa_spans_pos.split()
                q_offset = len(query_tokenized)
                for span in spans:
                    span = span.split(",")
                    span_start=int(span[0])
                    span_end=int(span[1])
                    got_start = False
                    got_end = False
                    for i,t in enumerate(doc_pos_tokenized):
                        if t.idx_end == None: # CLS tokens might return None
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

                if len(pos_qa_start) != len(pos_qa_end): # or len(pos_qa_start) != len(spans)
                    raise RuntimeError("error in qa span splitting: %s,%s" % (str(doc_pos_tokenized), qa_spans_pos)) 

            else:
                if self.qa_skip_noqa:
                    return None

            ret_instance["pos_qa_start"] = ArrayField(np.array(pos_qa_start),padding_value=-1,dtype=np.int64)
            ret_instance["pos_qa_end"] = ArrayField(np.array(pos_qa_end),padding_value=-1,dtype=np.int64)
            ret_instance["pos_qa_hasAnswer"] = ArrayField(np.array(pos_qa_hasAnswer),dtype=np.int64)

            
        if self.read_with_scores:
            ret_instance["pos_score"] = ArrayField(np.array(float(pos_score)))
            ret_instance["neg_score"] = ArrayField(np.array(float(neg_score)))


        if self.add_text_to_batch:
            ret_instance["query_text"] = MetadataField(query_sequence)
            ret_instance["doc_pos_text"] = MetadataField(doc_pos_sequence)
            ret_instance["doc_neg_text"] = MetadataField(doc_neg_sequence)
        
        return Instance(ret_instance)