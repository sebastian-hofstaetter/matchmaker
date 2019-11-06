from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict

import torch
import torch.nn as nn

from allennlp.nn.util import get_text_field_mask
                              
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention                          
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation

class DRMM(nn.Module):
    '''
    Paper: A Deep Relevance Matching Model for Ad-hoc Retrieval, Guo et al., CIKM'16
    only works with fixed word embeddings !
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 bin_count:int):
                 
        super(DRMM,self).__init__()

        self.word_embeddings = word_embeddings
        self.cosine_module = CosineMatrixAttention()

        self.bin_count = bin_count
        self.matching_classifier = FeedForward(input_dim=bin_count, num_layers=2, hidden_dims=[bin_count,1],activations=[Activation.by_name('tanh')(),Activation.by_name('tanh')()])
        self.query_gate = FeedForward(input_dim=self.word_embeddings.get_output_dim(), num_layers=2, hidden_dims=[self.word_embeddings.get_output_dim(),1],activations=[Activation.by_name('tanh')(),Activation.by_name('tanh')()])
        self.query_softmax = MaskedSoftmax()

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed
        if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
            # shape: (batch, query_max)
            query_pad_oov_mask = (query["tokens"] > 1).float()
            # shape: (batch, doc_max)
            document_pad_oov_mask = (document["tokens"] > 1).float()
        else: # == 3 (elmo characters per word)
            # shape: (batch, query_max)
            query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
            # shape: (batch, doc_max)
            document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        #
        # similarity matrix
        # -------------------------------------------------------

        # create sim matrix
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings).cpu()

        #
        # histogram & classfifier
        # ----------------------------------------------
        histogram_tensor = torch.empty((cosine_matrix.shape[0],cosine_matrix.shape[1],self.bin_count))
        for b in range(cosine_matrix.shape[0]):
            for q in range(cosine_matrix.shape[1]):
                histogram_tensor[b,q] = torch.histc(cosine_matrix[b,q], bins=self.bin_count, min=-1, max=1)

        histogram_tensor = histogram_tensor.to(device=query_embeddings.device)
        classified_matches_per_query = self.matching_classifier(torch.log1p(histogram_tensor)) # log1p is super important - lol just the opposite of knrm, does somebody understand the world??

        #
        # query gate 
        # ----------------------------------------------
        query_gates_raw = self.query_gate(query_embeddings)
        query_gates = self.query_softmax(query_gates_raw.squeeze(-1),query_pad_oov_mask).unsqueeze(-1)

        #
        # combine it all
        # ----------------------------------------------
        scores = torch.sum(classified_matches_per_query * query_gates,dim=1)

        return scores

    def get_param_stats(self):
        return "DRMM: -"


#from https://gist.github.com/kaniblu/94f3ede72d1651b087a561cf80b306ca thanks!
class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)
