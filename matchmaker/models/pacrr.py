from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict

import torch
import torch.nn as nn

from allennlp.nn.util import get_text_field_mask
                              
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention                          
from matchmaker.modules.masked_softmax import MaskedSoftmax

class PACRR(nn.Module):
    '''
    Paper: PACRR: A Position-Aware Neural IR Model for Relevance Matching, Hui et al., EMNLP'17

    Reference code (but in tensorflow):
    
    * first-hand: https://github.com/khui/copacrr/blob/master/models/pacrr.py
    
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return PACRR(unified_query_length=config["pacrr_unified_query_length"], 
                     unified_document_length=config["pacrr_unified_document_length"],
                     max_conv_kernel_size=config["pacrr_max_conv_kernel_size"],
                     conv_output_size=config["pacrr_conv_output_size"],
                     kmax_pooling_size=config["pacrr_kmax_pooling_size"])

    def __init__(self,

                 unified_query_length:int,
                 unified_document_length:int,

                 max_conv_kernel_size: int, # 2 to n
                 conv_output_size: int, # conv output channels

                 kmax_pooling_size: int): # per query k-max pooling
                 
        super(PACRR,self).__init__()

        self.cosine_module = CosineMatrixAttention()

        self.unified_query_length = unified_query_length
        self.unified_document_length = unified_document_length

        self.convolutions = []
        for i in range(2, max_conv_kernel_size + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad2d((0,i - 1,0, i - 1), 0), # this outputs [batch,1,unified_query_length + i - 1 ,unified_document_length + i - 1]
                    nn.Conv2d(kernel_size=i, in_channels=1, out_channels=conv_output_size), # this outputs [batch,32,unified_query_length,unified_document_length]
                    nn.MaxPool3d(kernel_size=(conv_output_size,1,1)) # this outputs [batch,1,unified_query_length,unified_document_length]
            ))
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        self.masked_softmax = MaskedSoftmax()
        self.kmax_pooling_size = kmax_pooling_size

        self.dense = nn.Linear(kmax_pooling_size * unified_query_length * max_conv_kernel_size, out_features=100, bias=True)
        self.dense2 = nn.Linear(100, out_features=10, bias=True)
        self.dense3 = nn.Linear(10, out_features=1, bias=False)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_idfs: torch.Tensor, document_idfs: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:

        #
        # similarity matrix
        # -------------------------------------------------------

        # create sim matrix
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        # shape: (batch, 1, query_max, doc_max) for the input of conv_2d
        cosine_matrix = cosine_matrix[:,None,:,:]

        #
        # duplicate cosine_matrix -> n-gram convolutions, then top-k pooling
        # ----------------------------------------------
        conv_results = []
        conv_results.append(torch.topk(cosine_matrix.squeeze(),k=self.kmax_pooling_size,sorted=True)[0])

        for conv in self.convolutions:
            cr = conv(cosine_matrix)
            cr_kmax_result = torch.topk(cr.squeeze(),k=self.kmax_pooling_size,sorted=True)[0]
            conv_results.append(cr_kmax_result)

        #
        # flatten all paths together & weight by query idf
        # -------------------------------------------------------
        
        per_query_results = torch.cat(conv_results,dim=-1)

        weigthed_per_query = per_query_results * self.masked_softmax(query_idfs, query_pad_oov_mask.unsqueeze(-1))

        all_flat = per_query_results.view(weigthed_per_query.shape[0],-1)


        #
        # dense layer
        # -------------------------------------------------------

        dense_out = F.relu(self.dense(all_flat))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)

        output = torch.squeeze(dense_out, 1)
        return output

    def get_param_stats(self):
        return "PACRR: / "
        #return "PACRR: dense3 weight: "+str(self.dense3.weight.data)+\
        #" dense2 weight: "+str(self.dense2.weight.data)+" b: "+str(self.dense2.bias.data) +\
        #" dense weight: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data)
