from typing import Dict, Iterator, List

import torch
import torch.nn as nn

class CollectionIndexerHead(nn.Module):
    '''
    Wraps a nn.module and calls forward_representation in forward (needed for multi-gpu use)
    '''

    def __init__(self,
                 neural_ir_model: nn.Module,
                 use_fp16=True):

        super(CollectionIndexerHead, self).__init__()

        self.neural_ir_model = neural_ir_model
        self.use_fp16 = use_fp16
        
    def forward(self, seq: Dict[str, torch.Tensor],title:torch.Tensor=None,) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if title is None:
                vectors = self.neural_ir_model.forward_representation(seq, sequence_type="doc_encode")
            else:
                vectors = self.neural_ir_model.forward_representation(seq, title, sequence_type="doc_encode")

            return vectors

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()

class QuerySearcherHead(nn.Module):
    '''
    Wraps a nn.module and calls forward_representation in forward (needed for multi-gpu use)
    '''

    def __init__(self,
                 neural_ir_model: nn.Module,
                 use_fp16=True):

        super(QuerySearcherHead, self).__init__()

        self.neural_ir_model = neural_ir_model
        self.use_fp16 = use_fp16

    def forward(self, seq: Dict[str, torch.Tensor],search_type="encode",document_enc=None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if search_type == "encode":
                vectors = self.neural_ir_model.forward_representation(seq, sequence_type="query_encode")
                return vectors
            elif search_type == "aggregate":
                scores = self.neural_ir_model.forward_aggregation(seq,document_enc)
                return scores

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()
