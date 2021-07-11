from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation

#
# prepare embedding tensors & paddings masks
# -------------------------------------------------------
def get_vectors_n_masks(word_embeddings,query,document):


    # shape: (batch, query_max,emb_dim)
    query_embeddings = word_embeddings({"tokens":{"tokens":query["tokens"]["tokens"]}})

    # shape: (batch, document_max,emb_dim)
    document_embeddings = word_embeddings({"tokens":{"tokens":document["tokens"]["tokens"]}})

    query = query["tokens"]
    document = document["tokens"]

    # we assume 1 is the unknown token, 0 is padding - both need to be removed
    if "mask" in query: #fasttext embedder
        query_mask = query["mask"].to(dtype=query_embeddings.dtype)
        # shape: (batch, doc_max)
        document_mask = document["mask"].to(dtype=query_embeddings.dtype)
    elif len(query["tokens"].shape) == 2: # (embedding lookup matrix)
        # shape: (batch, query_max)
        query_mask = (query["tokens"] > 0).to(dtype=query_embeddings.dtype)
        # shape: (batch, doc_max)
        document_mask = (document["tokens"] > 0).to(dtype=query_embeddings.dtype)
    else: # == 3 elmo
        # shape: (batch, query_max)
        query_mask = (torch.sum(query["tokens"],2) > 0).to(dtype=query_embeddings.dtype)
        # shape: (batch, doc_max)
        document_mask = (torch.sum(document["tokens"],2) > 0).to(dtype=query_embeddings.dtype)

    return query_embeddings,document_embeddings,query_mask,document_mask

def get_single_vectors_n_masks(word_embeddings,sequence):

    # shape: (batch, query_max,emb_dim)
    #sequence_embeddings = word_embeddings(sequence)
    sequence_embeddings = word_embeddings({"tokens":{"tokens":sequence["tokens"]["tokens"]}})
    sequence = sequence["tokens"]

    # we assume 1 is the unknown token, 0 is padding - both need to be removed
    if "mask" in sequence: #fasttext embedder
        sequence_mask = sequence["mask"].to(dtype=sequence_embeddings.dtype)
    elif len(sequence["tokens"].shape) == 2: # (embedding lookup matrix)
        sequence_mask = (sequence["tokens"] > 0).to(dtype=sequence_embeddings.dtype)
    else: # == 3 elmo
        sequence_mask = (torch.sum(sequence["tokens"],2) > 0).to(dtype=sequence_embeddings.dtype)

    return sequence_embeddings,sequence_mask


class NeuralIR_Encoder(nn.Module):
    '''
    Needs a neural IR model as paramter with a forward() that gets word vectors & masks as parameter
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 neural_ir_model: nn.Module):

        super(NeuralIR_Encoder, self).__init__()

        self.word_embeddings = word_embeddings
        self.neural_ir_model = neural_ir_model


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                use_fp16:bool = True, output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_embeddings,document_embeddings,query_mask,document_mask = get_vectors_n_masks(self.word_embeddings,query,document)
    
            inner_model_result = self.neural_ir_model.forward(query_embeddings, document_embeddings,
                                                              query_mask, document_mask, output_secondary_output)
    
            return inner_model_result
    
    def forward_representation(self, sequence: Dict[str, torch.Tensor], sequence_type:str) -> torch.Tensor:
        seq,mask = get_single_vectors_n_masks(self.word_embeddings,sequence)
        return self.neural_ir_model.forward_representation(seq,mask, sequence_type)

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()


class NeuralIR_Encoder_WithIds(nn.Module):
    '''
    Needs a neural IR model as paramter with a forward() that gets word vectors & masks & word ids as parameter
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 neural_ir_model: nn.Module):

        super(NeuralIR_Encoder_WithIds, self).__init__()

        self.word_embeddings = word_embeddings
        self.neural_ir_model = neural_ir_model


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                use_fp16:bool = True, output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_embeddings,document_embeddings,query_mask,document_mask = get_vectors_n_masks(self.word_embeddings,query,document)

            query_ids = query["tokens"]
            doc_ids = document["tokens"]

            inner_model_result = self.neural_ir_model.forward(query_embeddings, document_embeddings,
                                                              query_mask, document_mask,
                                                              query_ids, doc_ids, output_secondary_output)

            return inner_model_result
        
    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()
    
    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()


class NeuralIR_Encoder_WithIdfs(nn.Module):
    '''
    Needs a neural IR model as paramter with a forward() that gets word vectors & masks & word idfs as parameter
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 word_idfs: TextFieldEmbedder,
                 neural_ir_model: nn.Module):

        super(NeuralIR_Encoder_WithIdfs, self).__init__()

        self.word_embeddings = word_embeddings
        self.word_embeddings_idfs = word_idfs
        self.neural_ir_model = neural_ir_model


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                use_fp16:bool = True, output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_embeddings,document_embeddings,query_mask,document_mask = get_vectors_n_masks(self.word_embeddings,query,document)

            query_idfs = self.word_embeddings_idfs(query)
            doc_idfs = self.word_embeddings_idfs(document)

            inner_model_result = self.neural_ir_model.forward(query_embeddings, document_embeddings,
                                                              query_mask, document_mask,
                                                              query_idfs, doc_idfs, output_secondary_output)
            return inner_model_result

    def forward_representation(self, sequence: Dict[str, torch.Tensor], sequence_type:str) -> torch.Tensor:
        seq,mask = get_single_vectors_n_masks(self.word_embeddings,sequence)
        return self.neural_ir_model.forward_representation(seq,mask, sequence_type)

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()

class NeuralIR_Encoder_WithIdfs_PassThrough(nn.Module):
    '''
    Needs a neural IR model as paramter with a forward() that gets word ids & embedders as parameters (for full control)
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 word_idfs: TextFieldEmbedder,
                 neural_ir_model: nn.Module):

        super(NeuralIR_Encoder_WithIdfs_PassThrough, self).__init__()

        self.word_embeddings = word_embeddings
        self.word_embeddings_idfs = word_idfs
        self.neural_ir_model = neural_ir_model


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                use_fp16:bool = True, output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ
        with torch.cuda.amp.autocast(enabled=use_fp16):
            inner_model_result = self.neural_ir_model.forward(self.word_embeddings, self.word_embeddings_idfs,
                                                              query, document, output_secondary_output)
            return inner_model_result
    
    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()

class NeuralIR_Encoder_PassThrough(nn.Module):
    '''
    Needs a neural IR model as paramter with a forward() that gets word ids & embedders as parameters (for full control)
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 neural_ir_model: nn.Module):

        super(NeuralIR_Encoder_PassThrough, self).__init__()

        self.word_embeddings = word_embeddings
        self.neural_ir_model = neural_ir_model


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],title=None, output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ
        if title == None:
            return self.neural_ir_model.forward(self.word_embeddings, query, document, output_secondary_output)
        else:
            return self.neural_ir_model.forward(self.word_embeddings, query, document, title, output_secondary_output)

    def forward_representation(self, sequence: Dict[str, torch.Tensor],title: Dict[str, torch.Tensor]=None, sequence_type:str=None) -> torch.Tensor:
        if title == None:
            return self.neural_ir_model.forward_representation(self.word_embeddings,sequence, sequence_type)
        else:
            return self.neural_ir_model.forward_representation(self.word_embeddings,sequence,title, sequence_type)

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()