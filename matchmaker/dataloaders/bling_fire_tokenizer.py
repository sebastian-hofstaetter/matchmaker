from allennlp.data.tokenizers import Token
from blingfire import *
from typing import List


class BlingFireTokenizer():
    """
    basic tokenizer using bling fire library
    """

    def tokenize(self, sentence: str) -> List[Token]:
        return [Token(t) for t in text_to_words(sentence).split()]
