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
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import random
import copy
from collections import defaultdict
import torch 

@DatasetReader.register("mlm_seq_loader")
class MLMMaskedSequenceDatasetReader(DatasetReader):
    """
    Read a tsv file containing a single sequence <sequence_id>\t<sequence_string>
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_doc_length:int = -1,
                 min_doc_length:int = -1,
                 mlm_mask_whole_words:bool = True,
                 mask_probability:float = 0.1,
                 mlm_mask_replace_probability:float=0.5,
                 mlm_mask_random_probability:float=0.5,
                 bias_sampling_method="None",
                 bias_merge_alpha=0.5,
                 make_multiple_of=8,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self.max_seq_length = max_doc_length
        self.min_seq_length = min_doc_length

        self.max_title_length = 30
        self.min_title_length = -1
        self.mask_title = False

        self.token_type = "full"
        if type(tokenizer) == PretrainedTransformerTokenizer:
            self.token_type = "hf"
            self.padding_value = Token(text = "[PAD]", text_id=tokenizer.tokenizer.pad_token_id)
            self.mask_value = Token(text = "[MASK]", text_id=tokenizer.tokenizer.mask_token_id)
            self.cls_value = Token(text = "[CLS]", text_id=tokenizer.tokenizer.cls_token_id) # forgot to add <cls> to the vocabs
        else:
            self.padding_value = Token(text = "@@PADDING@@",text_id=0)
            self.mask_value = Token(text = "[MASK]",text_id=2)
        self.mask_probability = mask_probability
        self.mlm_mask_replace_probability = mlm_mask_replace_probability
        self.mlm_mask_random_probability = mlm_mask_random_probability
        self.mlm_mask_whole_words = mlm_mask_whole_words

        self.bias_sampling_method = bias_sampling_method
        self.bias_merge_alpha = bias_merge_alpha
        self.token_counter = np.ones(tokenizer.tokenizer.vocab_size,dtype=int)
        self.make_multiple_of = make_multiple_of

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) == 2:
                    seq_id, seq_text = line_parts
                    seq_title = None
                elif len(line_parts) == 3:
                    seq_id, seq_title, seq_text = line_parts
                    if seq_title == "" or seq_text=="":
                        continue
                else:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))

                yield self.text_to_instance(seq_id, seq_text, seq_title)

    @overrides
    def text_to_instance(self, seq_id:str, seq_text:str, seq_title:str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        seq_id_field = MetadataField(seq_id)

        seq_tokenized = self._tokenizer.tokenize(seq_text[:10_000])

        if self.max_seq_length > -1:
            seq_tokenized = seq_tokenized[:self.max_seq_length]
            #seq_tokenized_orig = seq_tokenized_orig[:self.max_seq_length]
            #mask_binary = mask_binary[:self.max_seq_length]
        if self.min_seq_length > -1 and len(seq_tokenized) < self.min_seq_length:
            seq_tokenized = seq_tokenized + [self.padding_value] * (self.min_seq_length - len(seq_tokenized))
            #seq_tokenized_orig = seq_tokenized_orig + [self.padding_value] * (self.min_seq_length - len(seq_tokenized_orig))
            #mask_binary = mask_binary + [0] * (self.min_seq_length - len(seq_tokenized))
        
        if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)


        seq_tokenized_orig = copy.deepcopy(seq_tokenized)
        mask_binary=[0] * len(seq_tokenized)

        suffix = "##" # self._tokenizer.tokenizer._parameters["suffix"]

        if self.token_type == "full":
            for i in range(len(seq_tokenized)):
                if random.uniform(0,1) < self.mask_probability:
                    if random.uniform(0,1) < self.mlm_mask_replace_probability:
                        seq_tokenized[i] = self.mask_value
                    mask_binary[i]= 1
                #else:
                #    mask_binary.append(0)
        else:

            tfs = np.ndarray(len(seq_tokenized_orig))
            for i,t in enumerate(seq_tokenized_orig):
                self.token_counter[t.text_id] += 1
                tfs[i] = self.token_counter[t.text_id]

            tf_class = tfs < np.median(tfs)

            if self.bias_sampling_method == "None":

                for i in range(len(seq_tokenized)):
                    replace_with_mask = False
                    replace_with_random = False
                    if i == 0 or (not self.mlm_mask_whole_words or seq_tokenized_orig[i-1].text.startswith(suffix)): # make sure to start masking at a word start
                        if random.uniform(0,1) < self.mask_probability:
                            if random.uniform(0,1) < self.mlm_mask_replace_probability:
                                replace_with_mask = True
                                seq_tokenized[i] = self.mask_value
                            elif random.uniform(0,1) < self.mlm_mask_random_probability:
                                replace_with_random = True
                                id_ = random.randint(0,self._tokenizer.tokenizer.vocab_size)
                                tok = self._tokenizer.tokenizer.convert_ids_to_tokens(id_)
                                seq_tokenized[i] = Token(text = tok, text_id=id_)

                            mask_binary[i] = 1
                            if self.mlm_mask_whole_words and not seq_tokenized_orig[i].text.startswith(suffix): # mask until end of full word 
                                for t in range(i+1,len(seq_tokenized)):
                                    if replace_with_mask == True:
                                        seq_tokenized[t] = self.mask_value
                                    elif replace_with_random == True:
                                        id_ = random.randint(0,self._tokenizer.tokenizer.vocab_size)
                                        tok = self._tokenizer.tokenizer.convert_ids_to_tokens(id_)
                                        seq_tokenized[t] = Token(text = tok, text_id=id_)                                    
                                    
                                    mask_binary[t] = 1
                                    if seq_tokenized_orig[t].text.startswith(suffix):
                                        break

            elif self.bias_sampling_method == "tf" or self.bias_sampling_method == "log-tf":

                if self.bias_sampling_method == "log-tf":
                    tfs = np.log2(tfs)

                probability = tfs.sum()/tfs
                probability /= probability.max()
                probability *= self.mask_probability
                #probability[probability < 0.0001] = 0.0001

                probability = probability * (self.mask_probability/(probability.mean()))
                probability[probability > 0.9] = 0.9

                #probability = probability * (self.mask_probability/(probability.mean()))
                #probability[probability > 0.9] = 0.9

                masks = torch.bernoulli(torch.from_numpy(probability))
                for i in range(len(seq_tokenized)):
                    if masks[i] == 1:
                        replace_with_mask = False
                        if random.uniform(0,1) < self.mlm_mask_replace_probability:
                            replace_with_mask = True
                            seq_tokenized[i] = self.mask_value
                        mask_binary[i] = 1

                        # opt 1 - previous tokens are part of the word -> mask them also
                        if i > 0 and not seq_tokenized_orig[i-1].text.endswith(suffix):
                            for t in list(range(0,i-1))[::-1]:
                                if replace_with_mask == True:
                                    seq_tokenized[t] = self.mask_value
                                mask_binary[t] = 1
                                if seq_tokenized_orig[t].text.endswith(suffix):
                                    break

                        # opt 2 - next tokens are part of the word -> mask them also
                        if not seq_tokenized_orig[i].text.endswith(suffix): # mask until end of full word 
                            for t in range(i+1,len(seq_tokenized)):
                                if replace_with_mask == True:
                                    seq_tokenized[t] = self.mask_value
                                mask_binary[t] = 1
                                if seq_tokenized_orig[t].text.endswith(suffix):
                                    break
        
        seq_field = TextField(seq_tokenized, self._token_indexers)
        seq_field_orig = TextField(seq_tokenized_orig, self._token_indexers)

        if seq_title != None:

            title_tokenized = self._tokenizer.tokenize(seq_title)

            if self.max_title_length > -1:
                title_tokenized = title_tokenized[:self.max_title_length]
            if self.min_title_length > -1 and len(title_tokenized) < self.min_title_length:
                title_tokenized = title_tokenized + [self.padding_value] * (self.min_title_length - len(title_tokenized))

            #if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            #    seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)
            
            title_tokenized.insert(0,self.cls_value)

            title_tokenized_masked = copy.deepcopy(title_tokenized)
            title_mask_binary=[0] * len(title_tokenized_masked)

            for i in range(len(title_tokenized_masked)):
                if random.uniform(0,1) < self.mask_probability:
                    if random.uniform(0,1) < self.mlm_mask_replace_probability:
                        replace_with_mask = True
                        title_tokenized_masked[i] = self.mask_value
                    title_mask_binary[i] = 1

            title_field = TextField(title_tokenized, self._token_indexers)
            title_field_masked = TextField(title_tokenized_masked, self._token_indexers)

            return Instance({
                "seq_id":seq_id_field,
                "title_tokens":title_field_masked,
                "title_tokens_original":title_field,
                "title_tokens_mask":ArrayField(np.array(title_mask_binary)),
                "seq_masked":ArrayField(np.array(mask_binary)),
                "seq_tf_info":ArrayField(np.array(tf_class)),
                "seq_tokens":seq_field,
                "seq_tokens_original":seq_field_orig})
        else:
            return Instance({
                "seq_id":seq_id_field,
                "seq_masked":ArrayField(np.array(mask_binary)),
                "seq_tf_info":ArrayField(np.array(tf_class)),
                "seq_tokens":seq_field,
                "seq_tokens_original":seq_field_orig})
