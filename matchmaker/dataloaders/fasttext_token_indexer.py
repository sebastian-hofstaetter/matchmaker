from typing import Dict, List

from overrides import overrides

from allennlp.common.checks import ConfigurationError
#from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
import numpy as np
import torch

class FastTextVocab():

    def __init__(self, mapping, shared_tensor,max_subwords) -> None:
        self.mapping = mapping
        self.data = shared_tensor
        self.max_subwords = max_subwords
        self.default = torch.zeros(1, dtype=torch.int)

    def get_subword_ids(self, word):
        #print(word)
        if word not in self.mapping:
            return self.default

        #print(self.mapping[word],self.data[self.mapping[word]])

        w_m = self.mapping[word]

        return self.data[w_m[0]:w_m[1]]

    @staticmethod
    def load_ids(file,max_subwords):
        mapping = {}
        data = []
        with open(file,"r",encoding="utf8") as in_file:
            for i,l in enumerate(in_file):
                l = l.rstrip().split()
                
                ids = [] # [0] * max_subwords
                for val in l[1:max_subwords]:
                    ids.append(int(val))

                mapping[l[0]] = (len(data),len(data) + len(ids))
                data.extend(ids)

        return mapping, torch.IntTensor(data)


class FastTextNGramIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of n-gram wordpiece ids to compute FastText representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``fasttext_grams``)
    """
    

    def __init__(self,
                 max_subwords,
                 namespace: str = 'fasttext_grams') -> None:
        super().__init__(0)
        self._namespace = namespace
        self.def_padding = torch.zeros(max_subwords, dtype=torch.int)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: FastTextVocab,
                          index_name: str) -> Dict[str, List[List[int]]]:

        #
        # unroll / offset the fasttext indices here -> in the embedding bag use the offsets 
        #

        token_offsets=[]
        offset=0       

        token_data_unrolled = []

        for token in tokens:

            if token.text_id == 0: # is the same as text=="@@PADDING@@":
                token_data_unrolled.append(0)
                token_offsets.append(offset)

            else:
                ids = vocabulary.get_subword_ids(token.text.lower())
                token_offsets.append(offset)
                offset += len(ids)
                token_data_unrolled.extend(ids.tolist())

        token_offsets.append(offset) # add the last one as well 

        return {index_name: token_data_unrolled,
                "offsets":token_offsets}

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        # pylint: disable=unused-argument
        return {}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    def _default_value_for_padding(self):
        return 0

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[List[int]]]:
        # pylint: disable=unused-argument

        all_tokens_padded = torch.LongTensor(pad_sequence_to_length(tokens["tokens"],desired_num_tokens["tokens"],default_value=0))
        offsets_padded = torch.LongTensor(pad_sequence_to_length(tokens["offsets"],desired_num_tokens["offsets"],default_value=tokens["offsets"][-1]))
        
        mask = ((np.array(tokens["offsets"][:-1]) - np.array(tokens["offsets"][1:])) != 0).astype(int).tolist()
        mask_padded = torch.LongTensor(pad_sequence_to_length(mask,desired_num_tokens["offsets"]-1,default_value=0))

        return {"tokens":all_tokens_padded,
                "offsets":offsets_padded,
                "mask":mask_padded}


def pad_sequence_to_length(sequence: List,
                           desired_length: int,
                           default_value: int = 0) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.
    
    if len(sequence) < desired_length:
        sequence.extend([default_value] * (desired_length - len(sequence)))
    # extends the sequence
    elif len(sequence) > desired_length:
        sequence = sequence[:desired_length]
    return sequence