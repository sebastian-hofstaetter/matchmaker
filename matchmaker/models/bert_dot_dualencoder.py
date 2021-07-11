from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import AutoModel
from transformers import DPRContextEncoder,DPRQuestionEncoder

class Bert_dot_dualencoder(nn.Module):
    """
    Huggingface LM (bert,distillbert,roberta,albert) model for dot-product (separated) q-d dot-product scoring 
    """
    def __init__(self,
                 bert_model_document: Union[str, AutoModel],
                 bert_model_query: Union[str, AutoModel],
                 return_vecs: bool = False,
                 trainable: bool = True) -> None:
        super().__init__()

        if isinstance(bert_model_document, str):
            if "facebook/dpr" in bert_model_document:
                self.bert_model_document = DPRContextEncoder.from_pretrained(bert_model_document)
            else:
                self.bert_model_document = AutoModel.from_pretrained(bert_model_document)
        else:
            self.bert_model_document = bert_model_document
        for p in self.bert_model_document.parameters():
            p.requires_grad = trainable

        if isinstance(bert_model_query, str):
            if "facebook/dpr" in bert_model_query:
                self.bert_model_query = DPRQuestionEncoder.from_pretrained(bert_model_query)
            else:
                self.bert_model_query = AutoModel.from_pretrained(bert_model_query)
        else:
            self.bert_model_query = bert_model_query
        for p in self.bert_model_query.parameters():
            p.requires_grad = trainable
        
        self.return_vecs = return_vecs

    def forward(self, query: Dict[str, torch.LongTensor], document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False):

        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_vecs = self.forward_representation(query,sequence_type="query_encode")
            document_vecs = self.forward_representation(document,sequence_type="doc_encode")

            score = torch.bmm(query_vecs.unsqueeze(dim=1), document_vecs.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.training and self.return_vecs:
                score = (score, query_vecs, document_vecs)

            if output_secondary_output:
                return score, {}
            return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type="doc_encode") -> torch.Tensor:
        
        model = self.bert_model_document
        if sequence_type == "query_encode":
            model = self.bert_model_query

        vectors = model(**tokens)[0][:,0,:]
        return vectors

    def get_param_stats(self):
        return "BERT_dot_dualencoder: / "
    def get_param_secondary(self):
        return {}