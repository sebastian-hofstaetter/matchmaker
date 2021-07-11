from typing import Dict, Union

import torch
from torch import nn as nn


class MaxPAdapter(nn.Module):
    """
    adapter for bert-tower and colbert for document chunking and returning the max-passage score
    """

    @staticmethod
    def from_config(config,inner_model, padding_idx):
        return MaxPAdapter(inner_model = inner_model,
                          chunk_size = config["idcm_chunk_size"],
                          overlap = config["idcm_overlap"],
                          padding_idx = padding_idx,
                          return_passage_scores_during_train = config["train_pairwise_distillation_on_passages"])

    def __init__(self,
                 inner_model: nn.Module,
                 chunk_size=50,
                 overlap=7,
                 padding_idx:int = 0,
                 return_passage_scores_during_train=False) -> None:
        super().__init__()

        self.is_teacher_model = False
        self.inner_model = inner_model

        #
        # chunking
        #
        self.padding_idx = padding_idx
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap

        self.return_passage_scores_during_train = return_passage_scores_during_train

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False):

        #
        # chunk documents & pack tensors
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
        orig_packed_indices = packed_indices.clone()

        ids_packed = chunked_ids_unrolled[packed_indices]
        mask_packed = (ids_packed != self.padding_idx)

        total_chunks=chunked_ids_unrolled.shape[0]

        packed_query_ids = query["input_ids"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["input_ids"].shape[1])[packed_indices]
        packed_query_mask = query["attention_mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["attention_mask"].shape[1])[packed_indices]

        inner_query = {"tokens":{"token_ids":packed_query_ids, "mask":packed_query_mask}}
        inner_document = {"tokens":{"token_ids":ids_packed, "mask":mask_packed}}
        
        #
        # run inner model on passages
        #
        inner_scores = self.inner_model(inner_query, inner_document, use_fp16)

        #
        # unpack scores & take max-p
        #
        scores_per_doc = torch.zeros((total_chunks,1), dtype=inner_scores.dtype, layout=inner_scores.layout, device=inner_scores.device)
        scores_per_doc[packed_indices] = inner_scores.unsqueeze(-1)
        scores_per_doc = scores_per_doc.reshape(batch_size,-1,)
        scores_per_doc_orig = scores_per_doc.clone()

        if (self.training or self.is_teacher_model) and self.return_passage_scores_during_train:
            return scores_per_doc

        scores_per_doc[scores_per_doc == 0] = -9000
        score = torch.max(scores_per_doc,dim=-1).values
        #score = torch.sort(scores_per_doc,descending=True,dim=-1).values
        #score[score <= -8900] = 0


        if output_secondary_output:
            return score, {
                "packed_indices": orig_packed_indices.view(batch_size,-1),
                "passage_scores":scores_per_doc_orig
            }
        else:
            return score    

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type=None) -> Dict[str, torch.Tensor]:
        
        if sequence_type == "doc_encode":
            
            document_ids = tokens["input_ids"][:,1:]
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
            orig_packed_indices = packed_indices.clone()

            ids_packed = chunked_ids_unrolled[packed_indices]
            mask_packed = (ids_packed != self.padding_idx)

            total_chunks=chunked_ids_unrolled.shape[0]

            inner_document = {"tokens":{"token_ids":ids_packed, "mask":mask_packed}}

            vecs = self.inner_model.forward_representation(inner_document)
            
            vecs_per_doc = torch.zeros((total_chunks,vecs.shape[-1]), dtype=vecs.dtype, layout=vecs.layout, device=vecs.device)
            vecs_per_doc[packed_indices] = vecs
            vecs_per_doc = vecs_per_doc.reshape(batch_size,-1,vecs.shape[-1])

            vecs = vecs_per_doc #* tokens["attention_mask"].unsqueeze(-1)
        
        if sequence_type == "query_encode":
            vecs = self.inner_model.forward_representation(tokens)
            vecs = vecs #* tokens["attention_mask"].unsqueeze(-1)

        return vecs

    def get_param_stats(self):
        return "MaxPAdapter: / "
    def get_param_secondary(self):
        return {}
