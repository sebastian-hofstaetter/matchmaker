from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
import math

class TKL_sigir20(nn.Module):
    '''
    TKL is a neural IR model for long documents
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TKL_sigir20(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
                     saturation_type= config["tk_saturation_type"],
                     )

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_ff_dim: int,
                 max_length,
                 use_pos_encoding,  
                 use_diff_posencoding,
                 saturation_type,
                 ):

        super(TKL_sigir20, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        self.re_use_encoding = True

        self.chunk_size = 40
        self.overlap = 5
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap
        
        self.sliding_window_size = 30
        self.top_k_chunks = 3

        self.use_idf_sat = saturation_type == "idf"
        self.use_embedding_sat = saturation_type == "embedding"
        self.use_linear_sat = saturation_type == "linear"
        self.use_log_sat = saturation_type == "log"

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = nn.Parameter(torch.cuda.FloatTensor(kernels_mu), requires_grad=False)#.view(1, 1, 1, n_kernels)
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False)#.view(1, 1, 1, n_kernels)
        #self.mu.data.requires_grad=True
        #self.sigma.data.requires_grad=True

        pos_f = self.get_positional_features(_embsize, 30) #max_timescale=100000
        pos_f.requires_grad = True
        self.positional_features_q = nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = True

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,2000+500+self.extended_chunk_size)[:,500:,:].clone() #max_timescale=100000
            pos_f.requires_grad = True
            self.positional_features_d = nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = True
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))
        self.mixer_sat = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        #self.emb_reducer = nn.Linear(_embsize, 300, bias=True)

        encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.saturation_linear = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.saturation_linear.bias, 100)
        torch.nn.init.uniform_(self.saturation_linear.weight, -0.014, 0.014)

        self.saturation_linear2 = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.saturation_linear2.bias, 100)
        torch.nn.init.uniform_(self.saturation_linear2.weight, -0.014, 0.014)

        self.saturation_linear3 = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.saturation_linear3.bias, 100)
        torch.nn.init.uniform_(self.saturation_linear3.weight, -0.014, 0.014)
        

        self.sat_normer = nn.LayerNorm(2,elementwise_affine=True)
        #self.sat_emb_reduce1 = nn.Linear(_embsize,_embsize, bias=False)
        self.sat_emb_reduce1 = nn.Linear(_embsize, 1, bias=False)
        #torch.nn.init.constant_(self.sat_emb_reduce1.bias, 2)

        self.kernel_mult = nn.Parameter(torch.full([4,1,1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        #self.length_normer = nn.Parameter(torch.full([1,1,1,1], 30, dtype=torch.float32, requires_grad=True))


        #self.max_chunks = int(max_length / self.chunk_size + 1)

        self.chunk_scoring = nn.Parameter(torch.full([1,self.top_k_chunks*5], 1, dtype=torch.float32, requires_grad=True))
        self.mixer_end = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                #query_idfs: torch.Tensor, document_idfs: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings_original = query_embeddings
        query_embeddings, query_embeddings_tf_output = self.forward_representation(query_embeddings, query_pad_oov_mask, self.positional_features_q[:,:query_embeddings.shape[1],:])


        if document_pad_oov_mask.shape[1] > self.overlap:
            needed_padding = self.extended_chunk_size - ((document_pad_oov_mask.shape[1] - self.overlap) % self.chunk_size)
        else:
            needed_padding = self.extended_chunk_size - self.overlap - document_pad_oov_mask.shape[1]

        document_embeddings = nn.functional.pad(document_embeddings,(0,0,self.overlap, needed_padding))
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(self.overlap, needed_padding))

        chunked_docs = document_embeddings.unfold(1,self.extended_chunk_size,self.chunk_size).transpose(-1,-2)
        chunked_pad = document_pad_oov_mask.unfold(1,self.extended_chunk_size,self.chunk_size)
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,self.extended_chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,self.extended_chunk_size)

        packed_indices = chunked_pad2[:,self.overlap:-self.overlap].sum(-1) != 0

        documents_packed = chunked_docs2[packed_indices]
        padding_packed = chunked_pad2[packed_indices]

        if self.re_use_encoding:
            document_pos_encoding = self.positional_features_d[:,:documents_packed.shape[1],:]
        else:
            document_pos_encoding = self.positional_features_d[:,:document_embeddings.shape[1],:]
            document_pos_encoding = document_pos_encoding.unfold(1,self.extended_chunk_size,self.chunk_size).transpose(-1,-2)
            document_pos_encoding = document_pos_encoding.squeeze(0)
            document_pos_encoding = document_pos_encoding.repeat(document_embeddings.shape[0],1,1)[packed_indices]

        documents_packed,_ = self.forward_representation(documents_packed, padding_packed, document_pos_encoding)

        documents_unique_again = documents_packed[:,self.overlap:-self.overlap,:]
        document_mask_packed_unique = padding_packed[:,self.overlap:-self.overlap]

        #
        # cosine matrix
        # -------------------------------------------------------
        packed_query_embeddings = query_embeddings.unsqueeze(1).expand(-1,chunk_pieces,-1,-1).reshape(-1,query_embeddings.shape[1],query_embeddings.shape[-1])[packed_indices]
        packed_query_mask = query_pad_oov_mask.unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query_embeddings.shape[1])[packed_indices]

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(packed_query_embeddings, documents_unique_again)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        cosine_matrix_extradim = cosine_matrix.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu.view(1, 1, 1, -1), 2) / (2 * torch.pow(self.sigma.view(1, 1, 1, -1), 2)))
        kernel_results_masked = raw_kernel_results * document_mask_packed_unique.unsqueeze(1).unsqueeze(-1)

        kerne_activations_per_doc = torch.zeros((chunked_docs2.shape[0],query_embeddings.shape[1],documents_unique_again.shape[1],kernel_results_masked.shape[-1]), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        kerne_activations_per_doc[packed_indices] = kernel_results_masked

        kerne_activations_per_doc = kerne_activations_per_doc.transpose(1,2).reshape(batch_size,-1,query_embeddings.shape[1],kernel_results_masked.shape[-1]).transpose(2,1)


        #
        # kernel-pooling
        # -------------------------------------------------------

        if kerne_activations_per_doc.shape[2] < self.sliding_window_size:
            kerne_activations_per_doc = nn.functional.pad(kerne_activations_per_doc,(0,0,0, self.sliding_window_size - kerne_activations_per_doc.shape[2]))

        unrolled_kernel_activations = kerne_activations_per_doc.unfold(2,self.sliding_window_size,2).transpose(-1,-2)
        unrolled_kernel_activation_lengths = torch.sum(unrolled_kernel_activations.sum(dim=-1) != 0,dim=-1)
        per_kernel_query = torch.sum(unrolled_kernel_activations, -2) 


        if self.use_idf_sat:
            sat_influencer = torch.cat([torch.relu(query_idfs.expand_as(unrolled_kernel_activation_lengths).unsqueeze(-1)),
                                        unrolled_kernel_activation_lengths.float().unsqueeze(-1)],dim=-1)

            sat1 = self.saturation_linear(sat_influencer)
            sat2 = 1 / self.saturation_linear2(sat_influencer)
            sat3 = self.saturation_linear3(sat_influencer)

            sat_per_kernel_query = sat1 * (torch.clamp(per_kernel_query, min=1e-10) ** sat2) - sat3

        elif self.use_embedding_sat:
            sat_influencer = torch.cat([self.sat_emb_reduce1(query_embeddings).expand_as(unrolled_kernel_activation_lengths).unsqueeze(-1),
                                        unrolled_kernel_activation_lengths.float().unsqueeze(-1)],dim=-1)

            sat_influencer = self.sat_normer(sat_influencer)

            sat1 = self.saturation_linear(sat_influencer)
            sat2 = 1 / self.saturation_linear2(sat_influencer)
            sat3 = self.saturation_linear3(sat_influencer)

            sat_per_kernel_query = sat1 * (torch.clamp(per_kernel_query, min=1e-10) ** sat2) - sat3

        elif self.use_linear_sat:
            sat_influencer = torch.cat([torch.relu(query_idfs.expand_as(unrolled_kernel_activation_lengths).unsqueeze(-1)),
                                        unrolled_kernel_activation_lengths.float().unsqueeze(-1)],dim=-1)

            sat1 = self.saturation_linear(sat_influencer)
            sat2 = self.saturation_linear2(sat_influencer)

            sat_per_kernel_query = sat1 * torch.clamp(per_kernel_query, min=1e-10) + sat2

        elif self.use_log_sat:
            sat_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_mult[0], min=1e-10))

        sat_per_kernel_query = sat_per_kernel_query * query_pad_oov_mask.unsqueeze(-1).unsqueeze(-1) * (unrolled_kernel_activation_lengths > 0).float().unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(sat_per_kernel_query, 1) 

        dense_out = self.dense(per_kernel)
        score = dense_out.squeeze(-1)

        if score.shape[1] < self.top_k_chunks:
            score = nn.functional.pad(score,(0, self.top_k_chunks - score.shape[1]))

        score[score == 0] = -9900
        orig_score = score

        #
        # argmax top-n hills
        # 
        top_non_overlapping_idx = torch.zeros((orig_score.shape[0],self.top_k_chunks), dtype=torch.long, device=orig_score.device) 
        max_per_region_score = orig_score.clone()

        r = torch.arange(max_per_region_score.shape[1],device=max_per_region_score.device)

        for c in range(0,self.top_k_chunks):
           
            best_index = torch.argmax(max_per_region_score,dim=1)
            top_non_overlapping_idx[:,c] = best_index
            region_pool = torch.abs(r - best_index.unsqueeze(-1)) < self.sliding_window_size / 2
            max_per_region_score[region_pool] = -10001 - c

       
        top_non_overlapping_idx_neighbors = torch.cat([top_non_overlapping_idx,top_non_overlapping_idx - 1,top_non_overlapping_idx + 1,top_non_overlapping_idx - 2,top_non_overlapping_idx + 2],dim=1)
        top_non_overlapping_idx_neighbors[top_non_overlapping_idx_neighbors < 0] = 0
        top_non_overlapping_idx_neighbors[top_non_overlapping_idx_neighbors >= orig_score.shape[1]] = orig_score.shape[1] - 1

        topk_indices_flat = (top_non_overlapping_idx_neighbors + torch.arange(0,orig_score.shape[0]*orig_score.shape[1],orig_score.shape[1],device=orig_score.device).unsqueeze(-1)).view(-1)
        top_k_non_overlapping = orig_score.view(-1).index_select(0,topk_indices_flat).view(top_non_overlapping_idx.shape[0],-1)
        top_k_non_overlapping[top_k_non_overlapping <= -9900] = 0

        orig_score[orig_score <= -9900] = 0

        score = (top_k_non_overlapping * self.chunk_scoring).sum(dim=1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            sat_influence_from_top_k = sat_influencer.transpose(1,2).reshape(-1,query_embeddings.shape[1],2).index_select(0,topk_indices_flat).view(top_non_overlapping_idx_neighbors.shape[0],top_non_overlapping_idx_neighbors.shape[1],query_embeddings.shape[1],2)
            return score, {"score":score,"orig_score":orig_score,"top_non_overlapping_idx":top_non_overlapping_idx,"orig_doc_len":document_pad_oov_mask.sum(dim=-1),"top_k_non_overlapping":top_k_non_overlapping,"sat_influence_from_top_k":sat_influence_from_top_k,
                           "total_chunks":chunked_docs2.shape[0],"packed_chunks":documents_packed.shape[0]}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor, positional_features=None) -> torch.Tensor:

        pos_sequence = sequence_embeddings
        if self.use_pos_encoding:
            if positional_features is None:
                positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]
            pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((pos_sequence).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
        sequence_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * sequence_embeddings_context) * sequence_mask.unsqueeze(-1)

        return sequence_embeddings,sequence_embeddings_context

    def get_positional_features(self,dimensions,
                                max_length,
                                min_timescale: float = 1.0,
                                max_timescale: float = 1.0e4):
        # pylint: disable=line-too-long
        """
        Implements the frequency-based positional encoding described
        in `Attention is all you Need
        <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

        Adds sinusoids of different frequencies to a ``Tensor``. A sinusoid of a
        different frequency and phase is added to each dimension of the input ``Tensor``.
        This allows the attention heads to use absolute and relative positions.

        The number of timescales is equal to hidden_dim / 2 within the range
        (min_timescale, max_timescale). For each timescale, the two sinusoidal
        signals sin(timestep / timescale) and cos(timestep / timescale) are
        generated and concatenated along the hidden_dim dimension.

        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The smallest timescale to use.
        max_timescale : ``float``, optional (default = 1.0e4)
            The largest timescale to use.

        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """
        timesteps=max_length
        hidden_dim = dimensions

        timestep_range = self.get_range_vector(timesteps, 0).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = self.get_range_vector(num_timescales, 0).data.float()

        log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
        inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
        return sinusoids.unsqueeze(0)

    def get_range_vector(self, size: int, device: int) -> torch.Tensor:
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\ # " bias: " +str(self.saturation_linear.bias.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        " self.chunk_scoring: " +str(self.chunk_scoring.data) +\
        " self.kernel_mult: " +str(self.kernel_mult.data) +\
        " self.saturation_linear: " +str(self.saturation_linear.weight.data) + " bias: " +str(self.saturation_linear.bias.data) +\
        " self.saturation_linear2: " +str(self.saturation_linear2.weight.data) + " bias: " +str(self.saturation_linear2.bias.data) +\
        " self.saturation_linear3: " +str(self.saturation_linear3.weight.data) + " bias: " +str(self.saturation_linear3.bias.data) +\
        "mixer: "+str(self.mixer.data) #+ "mixer_end: "+str(self.mixer_end.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,
                "saturation_linear_weight":self.saturation_linear.weight,
                "saturation_linear_bias":self.saturation_linear.bias,
                "saturation_linear2_weight":self.saturation_linear2.weight,
                "saturation_linear2_bias":self.saturation_linear2.bias,
                "saturation_linear3_weight":self.saturation_linear3.weight,
                "saturation_linear3_bias":self.saturation_linear3.bias,
                "chunk_scoring":self.chunk_scoring,
                "kernel_mult":self.kernel_mult,
                "mixer":self.mixer}