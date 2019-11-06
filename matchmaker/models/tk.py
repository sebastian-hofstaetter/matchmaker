from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder,MultiHeadSelfAttention
import math

class TK_v1(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v1(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"])

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int):

        super(TK_v1, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=att_proj_dim,
                 feedforward_hidden_dim=att_ff_dim,
                 num_layers=att_layer,
                 num_attention_heads=att_heads,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        #query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2) * query_pad_oov_mask.unsqueeze(-1)
        #document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2) * document_pad_oov_mask.unsqueeze(-1)
        query_embeddings = (self.mixer * query_embeddings + (1 - self.mixer) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = (self.mixer * document_embeddings + (1 - self.mixer) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"dense_mean_out":dense_mean_out,"per_kernel":per_kernel,
                           "per_kernel_mean":per_kernel_mean,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        seq_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)
        seq_embeddings_context = self.stacked_att(sequence_embeddings, sequence_mask)
        seq_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * seq_embeddings_context) * sequence_mask.unsqueeze(-1)
        return seq_embeddings

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +
        return "TK: dense w: "+str(self.dense.weight.data)+\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                "dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                "dense_comb_weight":self.dense_comb.weight, 
                "scaler":self.nn_scaler ,"mixer":self.mixer}



class TK_v2(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):

        ws = [20,30,50,80,100,120,150]
        max_windows = [math.ceil(config["max_doc_length"] / float(w)) for w in ws]

        return TK_v2(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =    config["tk_att_heads"],
                     att_layer =    config["tk_att_layer"],
                     att_proj_dim = config["tk_att_proj_dim"],
                     att_ff_dim =   config["tk_att_ff_dim"],
                     win_size = ws,
                     max_windows = max_windows)

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 win_size:int,
                 max_windows:int):

        super(TK_v2, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")


        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=att_proj_dim,
                 feedforward_hidden_dim=att_ff_dim,
                 num_layers=att_layer,
                 num_attention_heads=att_heads,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.nn_scaler = nn.ParameterList([nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True)) for w in win_size])

        self.kernel_weights = nn.ModuleList([nn.Linear(n_kernels, 1, bias=False) for w in win_size])
        
        self.window_size = win_size
        self.window_scorer = []
        for w in max_windows:
            l =  nn.Linear(w, 1, bias=False)
            torch.nn.init.constant_(l.weight, 1/w)
            self.window_scorer.append(l)

        self.window_scorer = nn.ModuleList(self.window_scorer)

        self.window_merger = nn.Linear(len(self.window_size), 1, bias=False)


    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        #query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2) * query_pad_oov_mask.unsqueeze(-1)
        #document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2) * document_pad_oov_mask.unsqueeze(-1)
        query_embeddings = (self.mixer * query_embeddings + (1 - self.mixer) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = (self.mixer * document_embeddings + (1 - self.mixer) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = torch.tanh(cosine_matrix * query_by_doc_mask)
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        individual_window_scores = []

        for i,window in enumerate(self.window_size):

            kernel_results_masked = nn.functional.pad(kernel_results_masked,(0,0,0,window - kernel_results_masked.shape[-2]%window)) 

            scoring_windows = kernel_results_masked.unfold(dimension=-2,size=window,step=window)

            scoring_windows = scoring_windows.transpose(-1,-2)
            #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

            per_kernel_query = torch.sum(scoring_windows, -2)
            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* 
            log_per_kernel_query_masked = log_per_kernel_query * (per_kernel_query.sum(dim=-1) != 0).unsqueeze(-1).float()
            #log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1).unsqueeze(-1) # make sure we mask out padding values
            per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

            window_scores = self.kernel_weights[i](per_kernel).squeeze(-1)

            window_scores_exp = torch.exp(window_scores * self.nn_scaler[i]) * (window_scores != 0).float()
            #window_scores_exp=window_scores
            if window_scores_exp.shape[-1] > self.window_scorer[i].in_features:
                window_scores_exp = window_scores_exp[:,:self.window_scorer[i].in_features]
            if window_scores_exp.shape[-1] < self.window_scorer[i].in_features:
                window_scores_exp = nn.functional.pad(window_scores_exp,(0,self.window_scorer[i].in_features - window_scores_exp.shape[-1])) 
  
            window_scores_exp = window_scores_exp.sort(dim=-1, descending=True)[0]

            individual_window_scores.append(self.window_scorer[i](window_scores_exp))
        #final_score = window_scores.sum(dim=-1) / (window_scores != 0).sum(dim=-1).float()

        

        final_window_score = self.window_merger(torch.cat(individual_window_scores,dim=1))
        score = torch.squeeze(final_window_score,1) #torch.tanh(dense_out), 1)
        if output_secondary_output:
            return score, {}
        return score


    def get_param_stats(self):
        return "tk_v2: "+\
            " ".join([" kernel_weight ("+str(self.window_size[i])+")"+str(w.weight.data) for i,w in enumerate(self.kernel_weights)])+"\n"+\
            " ".join([" nn_scaler ("+str(self.window_size[i])+")"+str(w.data) for i,w in enumerate(self.nn_scaler)])+"\n"+\
            " ".join([" window_scorer ("+str(self.window_size[i])+")"+str(w.weight.data) for i,w in enumerate(self.window_scorer)])+"\n"+\
            "mixer: "+str(self.mixer.data) + "window_merger: "+str(self.window_merger.weight.data)

    def get_param_secondary(self):
        return {#"dense_weight":self.dense.weight,"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,"dense_mean_bias":self.dense_mean.bias,
                "window_merger":self.window_merger.weight, 
                #"scaler: ":self.nn_scaler ,
                "mixer: ":self.mixer}