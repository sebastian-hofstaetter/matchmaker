from typing import Dict, Union
import logging

from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from matchmaker.dataloaders.transformer_tokenizer import *

from blingfire import *
import torch
import random


class IndependentTrainingDatasetReader(DatasetReader):
    """
    Read a tsv file containing training triple sequences

    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. 
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer=None,
                 token_indexers: Dict[str, TokenIndexer] = None,

                 max_doc_length: int = -1,
                 max_query_length: int = -1,
                 min_doc_length: int = -1,
                 min_query_length: int = -1,

                 data_augment: str = "none",

                 make_multiple_of: int = -1,

                 query_augment_mask_number: int = -1,

                 train_pairwise_distillation: bool = False,
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
        self.data_augment = data_augment

        if type(tokenizer) == FastTransformerTokenizer:
            self.token_type = "huggingface"
        else:
            self.token_type = "emb"
            self.padding_value = Token(text="@@PADDING@@", text_id=0)
            self.cls_value = Token(text="@@UNKNOWN@@", text_id=1)

        self.lucene_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of",
                                    "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])
        self.msmarco_top_50_stopwords = set([".", "the", ",", "of", "and", "a", "to", "in", "is", "-", "for", ")", "(", "or", "you", "that", "are", "on", "it", ":", "as", "with", "your",
                                            "'s", "from", "by", "be", "can", "an", "this", "1", "at", "have", "2", "not", "/", "was", "if", "$", "will", "i", "one", "which", "more", "has", "3", "but", "when", "what", "all"])
        self.msmarco_top_25_stopwords = set([".", "the", ",", "of", "and", "a", "to", "in", "is", "-", "for", ")",
                                            "(", "or", "you", "that", "are", "on", "it", ":", "as", "with", "your", "'s", "from"])
        self.use_stopwords = False
        self.make_multiple_of = make_multiple_of
        self.query_augment_mask_number = query_augment_mask_number

        self.max_title_length = 30
        self.min_title_length = -1

        self.read_with_scores = train_pairwise_distillation
        self.train_qa_spans = train_qa_spans
        self.add_text_to_batch = False

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            for line in self.shard_iterable(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                pos_score = None
                neg_score = None
                pos_score_passages = None
                neg_score_passages = None
                pos_title = None
                neg_title = None
                qa_spans_pos = None

                if not self.read_with_scores and not self.train_qa_spans:
                    if len(line_parts) == 3:
                        query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    elif len(line_parts) == 5:
                        query_sequence, pos_title, doc_pos_sequence, neg_title, doc_neg_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line))
                elif self.train_qa_spans:
                    if len(line_parts) == 4:
                        qa_spans_pos, query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line))
                else:
                    if len(line_parts) == 5:
                        pos_score, neg_score, query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    elif len(line_parts) == 7:
                        pos_score, pos_score_passages, neg_score, neg_score_passages, query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                    else:
                        raise ConfigurationError("Invalid line format: %s" % (line))

                inst = self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence, pos_title, neg_title, pos_score, neg_score, qa_spans_pos, pos_score_passages, neg_score_passages)
                if inst is not None:  # this should not happen (but just in case we don't break training)
                    yield inst

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str, pos_title, neg_title,
                         pos_score, neg_score, qa_spans_pos, pos_score_passages, neg_score_passages) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        def data_augment(aug_type, string):

            if aug_type == "shuffle_sent":
                doc_sequence = text_to_sentences(string).split("\n")
                random.shuffle(doc_sequence)
                doc_sequence = " ".join(doc_sequence)

            elif aug_type == "reverse_sent":
                doc_sequence = text_to_sentences(string).split("\n")
                doc_sequence = " ".join(doc_sequence[::-1])

            elif aug_type == "rotate":
                tokens = text_to_words(string).split()
                n = random.randint(0, len(tokens)-1)
                doc_sequence = " ".join(tokens[n:] + tokens[:n])

            elif aug_type == "none":
                doc_sequence = string
            else:
                raise Exception("wrong aug_type")

            return doc_sequence

        if self.data_augment != "none":
            if len(doc_pos_sequence) == 0 or len(doc_neg_sequence) == 0:
                return None
            doc_pos_sequence = data_augment(self.data_augment, doc_pos_sequence)
            doc_neg_sequence = data_augment(self.data_augment, doc_neg_sequence)

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

            # if self.make_multiple_of > -1 and len(query_tokenized) % self.make_multiple_of != 0:
            #    query_tokenized = query_tokenized + [self.padding_value] * (self.make_multiple_of - len(query_tokenized) % self.make_multiple_of)

            query_field = TextField(query_tokenized)

        if self.token_type == "huggingface":
            doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence, max_length=self.max_doc_length)
            doc_pos_field = PatchedTransformerTextField(**doc_pos_tokenized)

        else:
            doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence)
            if self.max_doc_length > -1:
                doc_pos_tokenized = doc_pos_tokenized[:self.max_doc_length]
            if self.min_doc_length > -1 and len(doc_pos_tokenized) < self.min_doc_length:
                doc_pos_tokenized = doc_pos_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_pos_tokenized))

            # if self.make_multiple_of > -1 and len(doc_pos_tokenized) % self.make_multiple_of != 0:
            #    doc_pos_tokenized = doc_pos_tokenized + [self.padding_value] * (self.make_multiple_of - len(doc_pos_tokenized) % self.make_multiple_of)

            # if self.use_stopwords:
            #    doc_pos_tokenized_filtered = []
            #    for t in doc_pos_tokenized:
            #        if t.text not in self.msmarco_top_25_stopwords:
            #            doc_pos_tokenized_filtered.append(t)
            #    doc_pos_tokenized = doc_pos_tokenized_filtered

            doc_pos_field = TextField(doc_pos_tokenized)

        if self.token_type == "huggingface":
            doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence, max_length=self.max_doc_length)
            doc_neg_field = PatchedTransformerTextField(**doc_neg_tokenized)
        else:
            doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence)

            if self.max_doc_length > -1:
                doc_neg_tokenized = doc_neg_tokenized[:self.max_doc_length]
            if self.min_doc_length > -1 and len(doc_neg_tokenized) < self.min_doc_length:
                doc_neg_tokenized = doc_neg_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_neg_tokenized))

            # if self.make_multiple_of > -1 and len(doc_neg_tokenized) % self.make_multiple_of != 0:
            #    doc_neg_tokenized = doc_neg_tokenized + [self.padding_value] * (self.make_multiple_of - len(doc_neg_tokenized) % self.make_multiple_of)

            # if self.use_stopwords:
            #    doc_neg_tokenized_filtered = []
            #    for t in doc_neg_tokenized:
            #        if t.text not in self.msmarco_top_25_stopwords:
            #            doc_neg_tokenized_filtered.append(t)
            #    doc_neg_tokenized = doc_neg_tokenized_filtered

            doc_neg_field = TextField(doc_neg_tokenized)

        if len(query_tokenized) == 0 or len(doc_neg_tokenized) == 0 or len(doc_pos_tokenized) == 0:
            return None

        ret_instance = {
            "query_tokens": query_field,
            "doc_pos_tokens": doc_pos_field,
            "doc_neg_tokens": doc_neg_field}

        if self.read_with_scores:
            ret_instance["pos_score"] = ArrayField(np.array(float(pos_score)))
            ret_instance["neg_score"] = ArrayField(np.array(float(neg_score)))
            if pos_score_passages != None:
                pos_score_passages = [float(f) for f in pos_score_passages.split()]
                ret_instance["pos_score_passages"] = ArrayField(np.array(pos_score_passages))
            if neg_score_passages != None:
                neg_score_passages = [float(f) for f in neg_score_passages.split()]
                ret_instance["neg_score_passages"] = ArrayField(np.array(neg_score_passages))

        if pos_title != None:

            if self.token_type == "huggingface":  # ugly fix, because tokenize() reads that var and no param
                self._tokenizer._max_length = self.max_title_length - 2

            pos_title_tokenized = self._tokenizer.tokenize(pos_title)
            neg_title_tokenized = self._tokenizer.tokenize(neg_title)

            if self.max_title_length > -1:
                pos_title_tokenized = pos_title_tokenized[:self.max_title_length]
                neg_title_tokenized = neg_title_tokenized[:self.max_title_length]
            if self.min_title_length > -1 and len(pos_title_tokenized) < self.min_title_length:
                pos_title_tokenized = pos_title_tokenized + [self.padding_value] * (self.min_title_length - len(pos_title_tokenized))
            if self.min_title_length > -1 and len(neg_title_tokenized) < self.min_title_length:
                neg_title_tokenized = neg_title_tokenized + [self.padding_value] * (self.min_title_length - len(neg_title_tokenized))

            # if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            #    seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)

            # pos_title_tokenized.insert(0,self.cls_value)
            # neg_title_tokenized.insert(0,self.cls_value)

            pos_title_field = TextField(pos_title_tokenized, self._token_indexers)
            neg_title_field = TextField(neg_title_tokenized, self._token_indexers)

            ret_instance["title_pos_tokens"] = pos_title_field
            ret_instance["title_neg_tokens"] = neg_title_field

        if self.add_text_to_batch:
            ret_instance["query_text"] = MetadataField(query_sequence)
            ret_instance["doc_pos_text"] = MetadataField(doc_pos_sequence)
            ret_instance["doc_neg_text"] = MetadataField(doc_neg_sequence)

        return Instance(ret_instance)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        if self.token_type != "huggingface":
            instance.fields["query_tokens"]._token_indexers = self._token_indexers  # type: ignore
            instance.fields["doc_pos_tokens"]._token_indexers = self._token_indexers  # type: ignore
            instance.fields["doc_neg_tokens"]._token_indexers = self._token_indexers  # type: ignore
