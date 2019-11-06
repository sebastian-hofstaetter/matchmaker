from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
import torch.nn.functional as F


class MV_LSTM(Model):
    '''
    Paper: A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations, Wan et al., AAAI'16

    Reference code (paper author): https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/mvlstm.py (but in tensorflow)

    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 lstm_hidden_dim: int,
                 top_k: int,
                 cuda_device: int) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        self.query_rep = nn.LSTM(self.word_embeddings.get_output_dim(),lstm_hidden_dim,batch_first=True,bidirectional=True)
        self.doc_rep = nn.LSTM(self.word_embeddings.get_output_dim(),lstm_hidden_dim,batch_first=True,bidirectional=True)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.top_k = top_k

        self.dense = nn.Linear(top_k, out_features=20, bias=True)
        self.dense2 = nn.Linear(20, out_features=20, bias=True)
        self.dense3 = nn.Linear(20, out_features=1, bias=False)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                query_length: torch.Tensor, document_length: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
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
        # conextualized rep (via lstms)
        # -------------------------------------------------------
        
        #hidden_d = torch.randn(())

        query_rep, hidden_q = self.query_rep(query_embeddings)
        document_rep, hidden_d = self.doc_rep(document_embeddings)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_rep, document_rep)

        #
        # topk pooling
        # -------------------------------------------------------
        
        cosine_flat = cosine_matrix.view(cosine_matrix.shape[0],-1)

        top_k_elments = torch.topk(cosine_flat,k=self.top_k,sorted=True)[0]

        ##
        ## "MLP" layer
        ## -------------------------------------------------------

        dense_out = F.relu(self.dense(top_k_elments))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)

        output = torch.squeeze(dense_out, 1)
        return output
