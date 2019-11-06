from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
import torch


class FastTextEmbeddingBag(EmbeddingBag):

    # embedding_matrix = fasttext input_matrix -> exported by generate_fasttext_weights.py (and loaded via numpy.load)
    def __init__(self, embedding_matrix, sparse=False,requires_grad=True,mode="sum"):
        embedding_matrix_shape = embedding_matrix.shape
        super().__init__(embedding_matrix_shape[0], embedding_matrix_shape[1], sparse=sparse,mode=mode)
        self.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.weight.data = self.weight.data * 10 # tk needs bigger values, for other models it doesn't matter
        #self.weight.data = self.weight.data / (self.weight.data.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        self.weight.requires_grad = requires_grad

    def get_output_dim(self):
        return self.weight.shape[1]

    # tokens,offsets is created via module/fasttext_token_embedder 
    # -> the ids must be created from the same model as the embedding_matrix via generate_fast_text_vocab_mapping.py
    def forward(self, tokens,offsets):

        #
        # tokens shape: (batch,max_token_len)
        # offsets shape: (batch,max_offsets_len i.e. word_length)
        #

        #
        # transform to 1d tensors
        #
        batch_size = offsets.shape[0]
        word_length = offsets.shape[1]
        tokens_flat = tokens.view(-1)
        
        # we need to add the token_max to the next line -> so we can flatten them to a 1d tensor
        offset_flat = (offsets + torch.arange(0,batch_size*tokens.shape[1],tokens.shape[1],device=offsets.device).unsqueeze(-1)).view(-1)
        
        #
        # run through embedding bag
        #
        out = super().forward(tokens_flat,offset_flat)
        out_batched = out.view(batch_size,word_length,-1)

        # remove the last dimension added by the offset process (only contains 0) 
        # - introduces problems if we need exact shapes
        # does not copy the data (!) so should be fast performance wise
        out_batched = out_batched[:,:word_length-1,:] 

        return out_batched
