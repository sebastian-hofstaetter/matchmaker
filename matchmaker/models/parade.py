from matchmaker.losses.lambdarank import LambdaLoss
from matchmaker.losses.soft_crossentropy import SoftCrossEntropy
from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import AutoModel


class Parade(nn.Module):
    '''
    Parade - Passage aggregation model: https://arxiv.org/pdf/2008.09093.pdf
    '''
    @staticmethod
    def from_config(config, padding_idx):
        return Parade(bert_model = config["bert_pretrained_model"],
                          trainable = config["bert_trainable"],
                          parade_aggregate_layers = config["parade_aggregate_layers"],
                          parade_aggregate_type = config["parade_aggregate_type"],
                          chunk_size = config["idcm_chunk_size"],
                          overlap = config["idcm_overlap"],
                          padding_idx = padding_idx)

    """
    document length model with local-self attion of bert model, and possible sampling of patches
    """
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 dropout: float = 0.0,
                 trainable: bool = True,
                 parade_aggregate_layers = 2,
                 parade_aggregate_type = "tf", # or max
                 chunk_size=50,
                 overlap=7,
                 padding_idx:int = 0) -> None:
        super().__init__()

        #
        # bert - scoring
        #
        if isinstance(bert_model, str):
            self.bert_model = AutoModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        for p in self.bert_model.parameters():
            p.requires_grad = trainable

        self._dropout = torch.nn.Dropout(p=dropout) 

        #
        # local self attention
        #
        self.padding_idx=padding_idx
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap

        #
        # aggregate
        #
        self.parade_aggregate_type = parade_aggregate_type

        if parade_aggregate_type == "tf":
            # appendix -> same config as bert
            encoder_layer = nn.TransformerEncoderLayer(self.bert_model.config.dim, self.bert_model.config.num_attention_heads,
                                                       dim_feedforward=self.bert_model.config.hidden_dim, dropout=self.bert_model.config.dropout)
            self.parade_aggregate_tf = nn.TransformerEncoder(encoder_layer, parade_aggregate_layers, norm=None)

        self.score_reduction = nn.Linear(self.bert_model.config.dim,1)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False):

        with torch.cuda.amp.autocast(enabled=use_fp16):

            #
            # patch up documents - local self attention
            #
            document_ids = document["input_ids"][:,1:]
            if document_ids.shape[1] > self.overlap:
                needed_padding = self.extended_chunk_size - (((document_ids.shape[1]) % self.chunk_size)  - self.overlap)
            else:
                needed_padding = self.extended_chunk_size - self.overlap - document_ids.shape[1]
            orig_doc_len = document_ids.shape[1]

            document_ids = nn.functional.pad(document_ids,(self.overlap, needed_padding),value=self.padding_idx)
            chunked_ids = document_ids.unfold(1,self.extended_chunk_size,self.chunk_size)

            batch_size = chunked_ids.shape[0]
            chunk_pieces = chunked_ids.shape[1]

            chunked_ids_unrolled=chunked_ids.reshape(-1,self.extended_chunk_size)
            packed_indices = (chunked_ids_unrolled[:,self.overlap:-self.overlap] != self.padding_idx).any(-1)
            ids_packed = chunked_ids_unrolled[packed_indices]
            mask_packed = (ids_packed != self.padding_idx)

            total_chunks=chunked_ids_unrolled.shape[0]

            packed_query_ids = query["input_ids"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["input_ids"].shape[1])[packed_indices]
            packed_query_mask = query["attention_mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["attention_mask"].shape[1])[packed_indices]

            #
            # packed bert scores
            #
            bert_vecs = self.forward_representation(torch.cat([packed_query_ids,ids_packed],dim=1),torch.cat([packed_query_mask,mask_packed],dim=1))

            vecs_per_doc = torch.zeros((total_chunks,bert_vecs.shape[-1]), dtype=bert_vecs.dtype, layout=bert_vecs.layout, device=bert_vecs.device)
            vecs_per_doc[packed_indices] = bert_vecs
            vecs_per_doc = vecs_per_doc.reshape(batch_size,-1,bert_vecs.shape[-1])

            # transformer aggregate
            if self.parade_aggregate_type == "tf":
                cls_vecs = self.bert_model.embeddings(query["input_ids"][:,0].unsqueeze(-1))
                agg_vec = self.parade_aggregate_tf(torch.cat([cls_vecs,vecs_per_doc],dim=1).transpose(1,0),
                                               src_key_padding_mask=torch.cat([torch.zeros((batch_size,1),device=packed_indices.device).bool(),
                                                                               ~packed_indices.view(batch_size,-1)],dim=1)
                                              ).transpose(1,0)[:,0]
             # max pooling aggregate
            else:
                agg_vec = torch.nn.functional.adaptive_max_pool2d(vecs_per_doc,output_size = (1,vecs_per_doc.shape[-1])).squeeze(1)
                
            score = self.score_reduction(agg_vec).squeeze(-1)

            if output_secondary_output:
                return score,{}
            else:
                return score    

    def forward_representation(self, ids,mask,type_ids=None) -> Dict[str, torch.Tensor]:
        
        if self.bert_model.base_model_prefix == 'distilbert': # diff input / output 
            pooled = self.bert_model(input_ids=ids,
                                     attention_mask=mask)[0][:,0,:]
        elif self.bert_model.base_model_prefix == 'longformer':
            _, pooled = self.bert_model(input_ids=ids,
                                        attention_mask=mask.long(),
                                        global_attention_mask = ((1-ids)*mask).long())
        elif self.bert_model.base_model_prefix == 'roberta': # no token type ids
            _, pooled = self.bert_model(input_ids=ids,
                                        attention_mask=mask)
        else:
            _, pooled = self.bert_model(input_ids=ids,
                                        token_type_ids=type_ids,
                                        attention_mask=mask)

        return pooled

    def get_param_stats(self):
        return "Parade: /"
    def get_param_secondary(self):
        return {}