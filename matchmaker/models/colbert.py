from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import PreTrainedModel,PretrainedConfig
from transformers import AutoModel
import math

class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True

class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    """
    
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"
    is_teacher_model = False # gets overriden by the dynamic teacher runner

    @staticmethod
    def from_config(config):
        cfg = ColBERTConfig()
        cfg.bert_model          = config["bert_pretrained_model"]
        cfg.compression_dim     = config["colbert_compression_dim"]
        cfg.return_vecs         = config.get("in_batch_negatives",False)
        cfg.trainable           = config["bert_trainable"]

        return ColBERT(cfg)
    
    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)
        
        self.return_vecs = cfg.return_vecs

        #if isinstance(bert_model, str):
        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)
        #else:
        #    self.bert_model = cfg.bert_model

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self._dropout = torch.nn.Dropout(p=cfg.dropout)
        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True, 
                output_secondary_output: bool = False) -> torch.Tensor:
                
        with torch.cuda.amp.autocast(enabled=use_fp16):

            #query_vecs = torch.nn.functional.normalize(self.forward_representation(query), p=2, dim=-1)
            #document_vecs = torch.nn.functional.normalize(self.forward_representation(document), p=2, dim=-1)

            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)

            score_per_term = torch.bmm(query_vecs, document_vecs.transpose(2,1))
            score_per_term[~(document["attention_mask"].bool()).unsqueeze(1).expand(-1,score_per_term.shape[1],-1)] = - 1000

            score = score_per_term.max(-1).values

            score[~(query["attention_mask"].bool())] = 0

            score = score.sum(-1)

            if self.is_teacher_model:
                return (score, query_vecs, document_vecs) #, score_per_term)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.return_vecs:
                score = (score, query_vecs, document_vecs)

            if output_secondary_output:
                return score, {}
            return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type=None) -> Dict[str, torch.Tensor]:
        
        vecs = self.bert_model(**tokens)[0]
        vecs = self.compressor(vecs)

        if sequence_type == "doc_encode" or sequence_type == "query_encode":
            vecs = vecs * tokens["attention_mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self,query_vecs, document_vecs):
        score = torch.bmm(query_vecs, document_vecs.transpose(2,1))
        #score[~document["attention_mask"].unsqueeze(1).expand(-1,score.shape[1],-1)] = - 10000
        
        score = score.max(-1).values
        
        #score[~query["attention_mask"]] = 0
        
        score = score.sum(-1)

        #if output_secondary_output:
        #    return score, {}
        return score

    def forward_inbatch_aggregation(self,query_vecs,query_mask, document_vecs, document_mask):
        #all_score_unrolled = []
        #for i in range(query_vecs.shape[0]):
        #    local_q = query_vecs[i]
        #    local_score = []
        #    for t in range(document_vecs.shape[0]):
        #    
        #        score = local_q @ document_vecs[t].T
        #        #score = torch.mm(local_q.view(-1,query_vecs.shape[-1]), document_vecs[t].view(-1,document_vecs.shape[-1]).transpose(-2,-1))\
        #        #             .view(1,query_vecs.shape[1],1,document_vecs.shape[1])
        #        #score=score.transpose(1,2)
        #        
        #        score[:,~document_mask[t]] = - 10000
        #        score = score.max(-1).values
        #        score[~query_mask[i]] = 0
        #        local_score.append(score.sum().unsqueeze(0))
        #
        #    #score = torch.cat(local_score,dim=0)
        #    #score[~query_mask[i].unsqueeze(1).expand(-1,score.shape[1],-1)] = 0
        #    all_score_unrolled.append(torch.cat(local_score,dim=0).unsqueeze(0))
        #
        #all_score_unrolled = torch.cat(all_score_unrolled,dim=0)

        #max_qs = 96
        #all_score = []
        #for i in range(math.ceil(query_vecs.shape[0] / max_qs)):
        #    local_q = query_vecs[i*max_qs:(i+1)*max_qs]
        #    score = torch.mm(local_q.view(-1,query_vecs.shape[-1]), document_vecs.view(-1,document_vecs.shape[-1]).transpose(-2,-1))\
        #                 .view(local_q.shape[0],query_vecs.shape[1],document_vecs.shape[0],document_vecs.shape[1])
        #    score=score.transpose(1,2)
##
        #    score[~document_mask.unsqueeze(0).unsqueeze(2).expand(score.shape[0],-1,score.shape[2],-1)] = - 10000
        #    score = score.max(-1).values
        #    score[~query_mask[i*max_qs:(i+1)*max_qs].unsqueeze(1).expand(-1,score.shape[1],-1)] = 0
        #    score = score.sum(-1)
        #    all_score.append(score)
##
        #all_score = torch.cat(all_score,dim=0)
        #return all_score

        score = torch.mm(query_vecs.view(-1,query_vecs.shape[-1]), document_vecs.view(-1,document_vecs.shape[-1]).transpose(-2,-1))\
                     .view(query_vecs.shape[0],query_vecs.shape[1],document_vecs.shape[0],document_vecs.shape[1])
        score=score.transpose(1,2)

        score[~(document_mask.bool()).unsqueeze(1).unsqueeze(1).expand(-1,score.shape[1],score.shape[2],-1)] = - 1000
        score = score.max(-1).values
        score[~(query_mask.bool()).unsqueeze(1).expand(-1,score.shape[1],-1)] = 0
        score = score.sum(-1)
        return score

    def get_param_stats(self):
        return "ColBERT: / "
    def get_param_secondary(self):
        return {}