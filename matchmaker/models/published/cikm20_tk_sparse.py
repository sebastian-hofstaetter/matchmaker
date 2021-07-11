from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
import math

class CIKM20_TK_Sparse(nn.Module):
    '''
    TK-Sparse is a neural re-ranking model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    -> allows for a sparsity vector to be leraned, that removes document terms from the scoring decision, 
        this sparsity vector needs to be forced by minimizing the L1 norm during training
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return CIKM20_TK_Sparse(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"])

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 max_length:int,
                 use_diff_posencoding:bool):

        super(CIKM20_TK_Sparse, self).__init__()      

        self.mixer_stop = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        #
        # Contextualization (pos. encoding + transformers)
        #
        self.use_diff_posencoding = use_diff_posencoding
        self.register_buffer("positional_features_q", self.get_positional_features(_embsize,max_length))
        if self.use_diff_posencoding == True:
            self.register_buffer("positional_features_d", self.get_positional_features(_embsize,max_length+500)[:,500:,:])
        else:
            self.register_buffer("positional_features_d", self.positional_features_q)

        encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        #
        # Cosine-match matrix 
        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights)
        #
        self.cosine_module = CosineMatrixAttention()

        #
        # Kernel pooling
        # with static - kernel size & magnitude buffers, and bin weights / scaling
        #
        n_kernels = len(kernels_mu)
        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # registered as buffers for multi-gpu training
        self.register_buffer("mu",nn.Parameter(torch.tensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels))
        self.register_buffer("sigma", nn.Parameter(torch.tensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels))

        self.kernel_bin_weights = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.kernel_bin_weights.weight, -0.014, 0.014) # inits taken from matchzoo
        
        # Newly added in TK-Sparse
        self.kernel_alpha_scaler = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))

        #
        # Sparse MLP
        #
        self.stop_word_reducer = nn.Linear(_embsize, 100, bias=True)
        self.stop_word_reducer2 = nn.Linear(100, 1, bias=True)
        torch.nn.init.constant_(self.stop_word_reducer2.bias, 1) # make sure we don't start in a broken state

    # Important during training, if stop_word_reducer2 only produces 0's, this re-animates 
    def reanimate(self,added_bias):
        self.stop_word_reducer2.bias.data += added_bias

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_mask: torch.Tensor, document_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings,_ = self.forward_representation(query_embeddings, query_mask,self.positional_features_q[:,:query_embeddings.shape[1],:])
        document_embeddings_orig = document_embeddings
        document_embeddings,document_embeddings_context = self.forward_representation(document_embeddings, document_mask,self.positional_features_d[:,:document_embeddings.shape[1],:])

        query_by_doc_mask = torch.bmm(query_mask.unsqueeze(-1), document_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        doc_lengths = torch.sum(document_mask, 1)

        #
        # cosine matrix
        # -------------------------------------------------------

        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask

        #
        # "kernel" part of kernel-pooling -> activate each elment in the match matrix by n_kernels
        # -> activations in range of 0,1 
        # -------------------------------------------------------

        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        #
        # "Sparse" part of TK-Sparse:
        # -> we learn a vector [bs,doc,1] that cancels out all kernel activations for 1 document term, if the vec. is 0 at this position
        # -> we apply the sparsity here, after the kernel pooling, because if we would apply it before (to the cosine-match-matrix) 
        #       the "0" kernel would count canceled words and we would not be able to 
        # -------------------------------------------------------

        document_embeddings_stop = (self.mixer_stop * document_embeddings_orig + (1 - self.mixer_stop) * document_embeddings_context)
        document_stop_words = torch.nn.functional.relu(self.stop_word_reducer2(torch.tanh(self.stop_word_reducer(document_embeddings_stop))).unsqueeze(1).squeeze(-1)) * document_mask.unsqueeze(1)

        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view * document_stop_words.unsqueeze(-1)

        #
        # "pooling" part of kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_alpha_scaler, min=1e-10)) 
        log_per_kernel_query_masked = log_per_kernel_query * query_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 
        score = self.kernel_bin_weights(per_kernel).squeeze(1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"per_kernel":per_kernel,
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked,"document_stop_words":document_stop_words}, document_stop_words
        else:
            return score, document_stop_words

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        if positional_features is None:
            positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        sequence_embeddings_context = self.contextualizer((sequence_embeddings + positional_features).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
        sequence_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * sequence_embeddings_context) * sequence_mask.unsqueeze(-1)

        return sequence_embeddings, sequence_embeddings_context


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

    def get_param_stats(self):
        return "TK: dense w: "+str(self.kernel_bin_weights.weight.data) +"self.kernel_alpha_scaler: "+str(self.kernel_alpha_scaler.data) +\
        "stop_word_reducer"+str(self.stop_word_reducer.weight.data)+str(self.stop_word_reducer.bias.data)+"stop_word_reducer2"+str(self.stop_word_reducer2.weight.data)+str(self.stop_word_reducer2.bias.data)+\
        "mixer: "+str(self.mixer.data)+"self.mixer_stop: "+str(self.mixer_stop.data)

    def get_param_secondary(self):
        return {"kernel_bin_weights":self.kernel_bin_weights.weight,
                "kernel_alpha_scaler":self.kernel_alpha_scaler,
                "mixer":self.mixer}
