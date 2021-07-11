from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          

class Conv_KNRM(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    third-hand reference: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py (tensorflow)
    https://github.com/thunlp/EntityDuetNeuralRanking/blob/master/baselines/CKNRM.py (pytorch)

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Conv_KNRM(word_embeddings_out_dim=word_embeddings_out_dim,
                         n_grams=config["conv_knrm_ngrams"], 
                         n_kernels=config["conv_knrm_kernels"],
                         conv_out_dim=config["conv_knrm_conv_out_dim"])

    def __init__(self,
                 word_embeddings_out_dim: int,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Conv_KNRM, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        self.convolutions = []
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings_out_dim, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=False) 

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_param_stats(self):
        return "CONV-KNRM: linear weight: "+str(self.dense.weight.data)
    def get_param_secondary(self):
        return {"kernel_weight":self.dense.weight}

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed
        #if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
#
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (query["tokens"] > 1).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (document["tokens"] > 1).float()
#
        #    # shape: (batch, query_max)
        #    query_pad_mask = (query["tokens"] > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_mask = (document["tokens"] > 0).float()
#
        #else: # == 3 (elmo characters per word)
        #    
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()
#
        ##query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
#
        ## shape: (batch, query_max,emb_dim)
        #query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        ## shape: (batch, document_max,emb_dim)
        #document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        query_pad_mask = query_pad_oov_mask
        document_pad_mask = document_pad_oov_mask

        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        query_results = []
        document_results = []

        for i,conv in enumerate(self.convolutions):
            query_conv = conv(query_embeddings_t).transpose(1, 2) 
            document_conv = conv(document_embeddings_t).transpose(1, 2)

            query_results.append(query_conv)
            document_results.append(document_conv)

        matched_results = []

        for i in range(len(query_results)):
            for t in range(len(query_results)):
                matched_results.append(self.forward_matrix_kernel_pooling(query_results[i], document_results[t], query_by_doc_mask, query_pad_mask))

        #
        # "Learning to rank" layer
        # -------------------------------------------------------

        all_grams = torch.cat(matched_results,1)

        dense_out = self.dense(all_grams)
        #tanh_out = torch.tanh(dense_out)

        output = torch.squeeze(dense_out, 1)
        if output_secondary_output:
            return output, {}
        return output

    #
    # create a match matrix between query & document terms
    #
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        
        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        return per_kernel

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma