from typing import Dict, Union,List

from overrides import overrides
import torch

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from transformers import AutoModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from matchmaker.modules.bert_parts import *

class BertEmbeddingTokenEmbedder(TokenEmbedder):
    """

    Take only bert embeddings + positional embeddings (no bert encoding model) 
    for use in other models

    Parameters
    ----------
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    """
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 pos_embeddings = False,
                 keep_layers = False,
                 trainable: bool = True) -> None:
        super(BertEmbeddingTokenEmbedder, self).__init__()

        if isinstance(bert_model, str):
            bert_model = AutoModel.from_pretrained(bert_model)

        self.output_dim = bert_model.config.hidden_size

        if not pos_embeddings:
            self.bert_embeddings = BertNoPosEmbeddings(bert_model.embeddings.word_embeddings,bert_model.embeddings.token_type_embeddings,
                                                        bert_model.embeddings.LayerNorm,bert_model.embeddings.dropout)
        else:
            self.bert_embeddings = bert_model.embeddings

        if keep_layers:
            self.bert_layers = bert_model.encoder
            self.bert_pos_enc = bert_model.embeddings.position_embeddings

        del bert_model

        self.bert_embeddings.requires_grad = trainable

    def get_bert_layers(self):
        b = (self.bert_layers,self.bert_pos_enc)
        del self.bert_layers
        del self.bert_pos_enc
        return b

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                tokens: torch.LongTensor,
                offsets: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None) -> torch.Tensor:
        # token_type_ids are ignored

        embeddings = self.bert_embeddings.forward(tokens)

        return embeddings

from allennlp.data.token_indexers import PretrainedTransformerIndexer

class PretrainedBertIndexerNoSpecialTokens(PretrainedTransformerIndexer):

    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.
        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(
        self,
        pretrained_model: str,
        use_starting_offsets: bool = False,
        do_lowercase: bool = True,
        never_lowercase: List[str] = None,
        max_pieces: int = 512,
        truncate_long_sequences: bool = True,
    ) -> None:

        bert_tokenizer = PretrainedTransformerTokenizer(pretrained_model, do_lower_case=do_lowercase)
        super().__init__(
            vocab=bert_tokenizer.vocab,
            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
            namespace="bert",
            use_starting_offsets=use_starting_offsets,
            max_pieces=max_pieces,
            do_lowercase=do_lowercase,
            never_lowercase=never_lowercase,
            start_tokens=[],
            end_tokens=[],
            separator_token="[SEP]",
            truncate_long_sequences=truncate_long_sequences,
        )

    def __eq__(self, other):
        if isinstance(other, PretrainedBertIndexerNoSpecialTokens):
            for key in self.__dict__:
                if key == "wordpiece_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented