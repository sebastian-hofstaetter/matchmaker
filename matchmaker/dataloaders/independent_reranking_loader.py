from typing import Dict

from overrides import overrides
import numpy as np
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from matchmaker.dataloaders.transformer_tokenizer import *

class IndependentReRankingDatasetReader(DatasetReader):
    """
    Read a tsv file containing re-ranking candidate sequences
    
    Expected format for each input line: <query_id>\t<doc_id>\t<query_sequence_string>\t<doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_tokens: ``TextField`` 


    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. 
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 min_doc_length:int = -1,
                 min_query_length:int = -1,

                 query_augment_mask_number:int = -1,
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

        if type(tokenizer) == FastTransformerTokenizer:
            self.token_type = "huggingface"
        else:
            self.token_type = "emb"
            self.padding_value = Token(text = "@@PADDING@@",text_id=0)
            self.cls_value = Token(text = "@@UNKNOWN@@",text_id=1)

        self.lucene_stopwords=set(["a", "an", "and", "are", "as", "at", "be", "but", "by","for", "if", "in", "into", "is", "it","no", "not", "of", "on", "or", "such","that", "the", "their", "then", "there", "these","they", "this", "to", "was", "will", "with"])
        self.msmarco_top_50_stopwords=set([".","the",",","of","and","a","to","in","is","-","for",")","(","or","you","that","are","on","it",":","as","with","your","'s","from","by","be","can","an","this","1","at","have","2","not","/","was","if","$","will","i","one","which","more","has","3","but","when","what","all"])
        self.msmarco_top_25_stopwords=set([".","the",",","of","and","a","to","in","is","-","for",")","(","or","you","that","are","on","it",":","as","with","your","'s","from"])
        self.use_stopwords = False
        self.query_augment_mask_number = query_augment_mask_number

        self.max_title_length = 30
        self.min_title_length = -1
        self.train_qa_spans = train_qa_spans

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line in self.shard_iterable(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) == 4:
                    query_id, doc_id, query_sequence, doc_sequence = line_parts
                    doc_title = None
                elif len(line_parts) == 5:
                    query_id, doc_id, query_sequence, doc_title,doc_sequence = line_parts
                else:
                    raise ConfigurationError("Invalid line format: %s" % (line))

                yield self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence,doc_title)

    @overrides
    def text_to_instance(self, query_id:str, doc_id:str, query_sequence: str, doc_sequence: str,doc_title) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        query_id_field = MetadataField(query_id)
        doc_id_field = MetadataField(doc_id)

        if self.token_type == "huggingface":
            query_tokenized = self._tokenizer.tokenize(query_sequence, max_length=self.max_query_length)

            if self.query_augment_mask_number > -1:
                query_tokenized["input_ids"] = torch.cat([torch.nn.functional.pad(query_tokenized["input_ids"][:-1],
                                                                                  (0, self.query_augment_mask_number),
                                                                                  value=self._tokenizer._tokenizer.mask_token_id), query_tokenized["input_ids"][-1].unsqueeze(0)])
                query_tokenized["attention_mask"] = torch.nn.functional.pad(query_tokenized["attention_mask"],
                                                                            (0, self.query_augment_mask_number),
                                                                            value=1)

            query_field = PatchedTransformerTextField(**query_tokenized)

        else:
            query_tokenized = self._tokenizer.tokenize(query_sequence)

            if self.max_query_length > -1:
                query_tokenized = query_tokenized[:self.max_query_length]
            if self.min_query_length > -1 and len(query_tokenized) < self.min_query_length:
                query_tokenized = query_tokenized + [self.padding_value] * (self.min_query_length - len(query_tokenized))

            query_field = TextField(query_tokenized)

        if self.token_type == "huggingface":
            doc_tokenized = self._tokenizer.tokenize(doc_sequence, max_length=self.max_doc_length)
            doc_field = PatchedTransformerTextField(**doc_tokenized)

        else:
            doc_tokenized = self._tokenizer.tokenize(doc_sequence)
            if self.max_doc_length > -1:
                doc_tokenized = doc_tokenized[:self.max_doc_length]
            if self.min_doc_length > -1 and len(doc_tokenized) < self.min_doc_length:
                doc_tokenized = doc_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_tokenized))

            doc_field = TextField(doc_tokenized)

        if doc_title != None:
            
            if self.token_type == "huggingface": # ugly fix, because tokenize() reads that var and no param 
                self._tokenizer._max_length = self.max_title_length - 2

            title_tokenized = self._tokenizer.tokenize(doc_title)

            if self.max_title_length > -1:
                title_tokenized = title_tokenized[:self.max_title_length]
            if self.min_title_length > -1 and len(title_tokenized) < self.min_title_length:
                title_tokenized = title_tokenized + [self.padding_value] * (self.min_title_length - len(title_tokenized))

            #if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            #    seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)
            
            title_tokenized.insert(0,self.cls_value)

            title_field = TextField(title_tokenized)

            return Instance({
                "query_id":query_id_field,
                "doc_id":doc_id_field,
                "query_tokens":query_field,
                "title_tokens":title_field,
                "doc_tokens":doc_field})

        return Instance({
            "query_id":query_id_field,
            "doc_id":doc_id_field,
            "query_tokens":query_field,
            "doc_tokens":doc_field})

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        if self.token_type != "huggingface":
            instance.fields["query_tokens"]._token_indexers = self._token_indexers  # type: ignore
            instance.fields["doc_tokens"]._token_indexers = self._token_indexers  # type: ignore
