from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


from allennlp.nn.util import get_text_field_mask
                              
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention                          


class MatchPyramid(nn.Module):
    '''
    Paper: Text Matching as Image Recognition, Pang et al., AAAI'16

    Reference code (but in tensorflow):
    
    * first-hand: https://github.com/pl8787/MatchPyramid-TensorFlow/blob/master/model/model_mp.py
    
    * somewhat-third-hand reference: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/matchpyramid.py

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return MatchPyramid(conv_output_size=config["match_pyramid_conv_output_size"], 
                            conv_kernel_size=config["match_pyramid_conv_kernel_size"],
                            adaptive_pooling_size=config["match_pyramid_adaptive_pooling_size"])

    def __init__(self,
                 conv_output_size: List[int],
                 conv_kernel_size: List[Tuple[int,int]],
                 adaptive_pooling_size: List[Tuple[int,int]]):

        super(MatchPyramid, self).__init__()

        self.cosine_module = CosineMatrixAttention()

        if len(conv_output_size) != len(conv_kernel_size) or len(conv_output_size) != len(adaptive_pooling_size):
            raise Exception("conv_output_size, conv_kernel_size, adaptive_pooling_size must have the same length")

        conv_layer_dict = OrderedDict()
        last_channel_out = 1
        for i in range(len(conv_output_size)):
            conv_layer_dict["pad " +str(i)] = nn.ConstantPad2d((0,conv_kernel_size[i][0] - 1,0, conv_kernel_size[i][1] - 1), 0)
            conv_layer_dict["conv "+str(i)] = nn.Conv2d(kernel_size=conv_kernel_size[i], in_channels=last_channel_out, out_channels=conv_output_size[i])
            conv_layer_dict["relu "+str(i)] = nn.ReLU()
            conv_layer_dict["pool "+str(i)] = nn.AdaptiveMaxPool2d(adaptive_pooling_size[i])
            last_channel_out = conv_output_size[i]

        self.conv_layers = nn.Sequential(conv_layer_dict)

        self.dense = nn.Linear(conv_output_size[-1] * adaptive_pooling_size[-1][0] * adaptive_pooling_size[-1][1], out_features=100, bias=True)
        self.dense2 = nn.Linear(100, out_features=10, bias=True)
        self.dense3 = nn.Linear(10, out_features=1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        #torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:

        #
        # similarity matrix
        # -------------------------------------------------------

        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        # shape: (batch, 1, query_max, doc_max) for the input of conv_2d
        cosine_matrix = cosine_matrix[:,None,:,:]

        #
        # convolution
        # -------------------------------------------------------
        # shape: (batch, conv_output_size, query_max, doc_max) 

        conv_result = self.conv_layers(cosine_matrix) 

        #
        # dynamic pooling
        # -------------------------------------------------------
        
        # flatten the output of dynamic pooling

        # shape: (batch, conv_output_size * pool_h * pool_w) 
        conv_result_flat = conv_result.view(conv_result.size(0), -1)

        #conv_result_flat = self.dropout(conv_result_flat)

        #
        # Learning to rank layer
        # -------------------------------------------------------
        dense_out = F.relu(self.dense(conv_result_flat))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)
        #tanh_out = torch.tanh(dense_out)

        output = torch.squeeze(dense_out, 1)
        if output_secondary_output:
            return output, {}
        return output

    def get_param_stats(self):
        return "MP: / "
    def get_param_secondary(self):
        return {}
