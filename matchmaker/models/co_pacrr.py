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

class CO_PACRR(nn.Module):
    '''
    Paper: Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, Hui et al., WSDM'18

    Reference code (but in tensorflow):
    
    * first-hand: https://github.com/khui/copacrr/blob/master/models/pacrr.py
    
    differences to pacrr: 
      * (1) context vector (query avg, document rolling window avg pool)
      * (2) cascade k-max pooling
      * (3) shuffling query terms at the end
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return CO_PACRR(unified_query_length=config["pacrr_unified_query_length"], 
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
                 
        super(CO_PACRR,self).__init__()

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

        context_pool_size = 6
        self.doc_context_pool = nn.Sequential(
                                    nn.ConstantPad1d((0,context_pool_size - 1),0),
                                    nn.AvgPool1d(kernel_size=context_pool_size,stride=1))

        self.masked_softmax = MaskedSoftmax()
        self.kmax_pooling_size = kmax_pooling_size

        kmax_pooling_view_percent = [0.25,0.5,0.75,1]
        self.kmax_pooling_views = [int(unified_document_length * x) for x in kmax_pooling_view_percent]

        self.dense = nn.Linear(len(self.kmax_pooling_views) * 2 * kmax_pooling_size * unified_query_length * max_conv_kernel_size, out_features=100, bias=True)
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
        # generate query and doc contexts
        # -------------------------------------------------------

        query_context = torch.mean(query_embeddings,dim=1)
        document_context = self.doc_context_pool(document_embeddings.transpose(1,2)).transpose(1,2) 

        cosine_matrix_context = self.cosine_module.forward(query_context.unsqueeze(dim=1),document_context).squeeze(1)

        #
        # duplicate cosine_matrix -> n-gram convolutions, then top-k pooling
        # ----------------------------------------------
        conv_results = []

        #
        # 1x1 cosine matrix (extra without convolutions)
        #
        
        cr_kmax_result = [[],[]]
        
        for view_size in self.kmax_pooling_views:
            val,idx = torch.topk(cosine_matrix.squeeze(dim=1)[:,:,0:view_size],k=self.kmax_pooling_size,sorted=True)
            cr_kmax_result[0].append(val)
            cr_kmax_result[1].append(idx)

        cr_kmax_result[0] = torch.cat(cr_kmax_result[0],dim=-1)
        cr_kmax_result[1] = torch.cat(cr_kmax_result[1],dim=-1)

        # incorporate context sims here, by selecting them from the kmax of the non-context sims
        flat_context = cosine_matrix_context.view(-1)
        index_offset = cr_kmax_result[1] + torch.arange(0,cr_kmax_result[1].shape[0]*cosine_matrix_context.shape[1],cosine_matrix_context.shape[1],device=cr_kmax_result[1].device).unsqueeze(-1).unsqueeze(-1)
        selected_context = flat_context.index_select(dim=0,index=index_offset.view(-1)).view(cr_kmax_result[1].shape[0],cr_kmax_result[1].shape[1],-1)
        conv_results.append(torch.cat([cr_kmax_result[0],selected_context],dim=2))

        #
        # nxn n-gram cosine matrices
        #
        for conv in self.convolutions:
            cr = conv(cosine_matrix)

            #
            # (2) take the kmax at multiple views of the cosine matrix - always starting
            #

            cr_kmax_result = [[],[]]
            for view_size in self.kmax_pooling_views:
                val,idx = torch.topk(cr.squeeze(dim=1)[:,:,0:view_size],k=self.kmax_pooling_size,sorted=True)
                cr_kmax_result[0].append(val)
                cr_kmax_result[1].append(idx)
            cr_kmax_result[0] = torch.cat(cr_kmax_result[0],dim=-1)
            cr_kmax_result[1] = torch.cat(cr_kmax_result[1],dim=-1)

            #
            # (1) incorporate context sims here, by selecting them from the kmax of the non-context sims
            #
            flat_context = cosine_matrix_context.view(-1)
            index_offset = cr_kmax_result[1] + torch.arange(0,cr_kmax_result[1].shape[0]*cosine_matrix_context.shape[1],cosine_matrix_context.shape[1],device=cr_kmax_result[1].device).unsqueeze(-1).unsqueeze(-1)
            selected_context = flat_context.index_select(dim=0,index=index_offset.view(-1)).view(cr_kmax_result[1].shape[0],cr_kmax_result[1].shape[1],-1)
            conv_results.append(torch.cat([cr_kmax_result[0],selected_context],dim=2))

        #
        # flatten all paths together & weight by query idf
        # -------------------------------------------------------
        
        per_query_results = torch.cat(conv_results,dim=-1)

        weighted_per_query = per_query_results * self.masked_softmax(query_idfs, query_pad_oov_mask.unsqueeze(-1))
        
        #
        # (3) shuffle component
        #
        if self.training:
            weighted_per_query = weighted_per_query[:,torch.randperm(weighted_per_query.shape[1]),:]

        all_flat = per_query_results.view(weighted_per_query.shape[0],-1)


        #
        # dense layer
        # -------------------------------------------------------

        dense_out = F.relu(self.dense(all_flat))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)

        output = torch.squeeze(dense_out, 1)
        if output_secondary_output:
            return output, {}
        return output

    def get_param_stats(self):
        return "CO-PACRR: / "

    def get_param_secondary(self):
        return {}