from matchmaker.losses.lambdarank import LambdaLoss
from matchmaker.losses.soft_crossentropy import SoftCrossEntropy
from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import AutoModel


class IDCM(nn.Module):
    '''
    IDCM is a neural re-ranking model for long documents, it creates an intra-document cascade between a fast (CK) and a slow module (BERT_Cat)
    '''

    @staticmethod
    def from_config(config, padding_idx):
        return IDCM(bert_model = config["bert_pretrained_model"],
                    trainable = config["bert_trainable"],
                    sample_train_type = config["idcm_sample_train_type"],
                    sample_n = config["idcm_sample_n"],
                    sample_context =config["idcm_sample_context"],
                    top_k_chunks = config["idcm_top_k_chunks"],
                    chunk_size = config["idcm_chunk_size"],
                    overlap = config["idcm_overlap"],
                    padding_idx=padding_idx)

    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 dropout: float = 0.0,
                 trainable: bool = True,
                 sample_train_type = "lambdaloss",
                 # how many passages get scored by BERT 
                 sample_n = 1,
                 # type of fast module
                 sample_context = "ck",
                 # how many passages to take from bert to create the final score (usually the same as sample_n, but could be set to 1 for max-p)
                 top_k_chunks=3,
                 # window size
                 chunk_size=50,
                 # left and right overlap (added to each window)
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

        #
        # final scoring (combination of bert scores)
        #
        self._classification_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)
        self.top_k_chunks = top_k_chunks
        self.top_k_scoring = nn.Parameter(torch.full([1,self.top_k_chunks], 1, dtype=torch.float32, requires_grad=True))

        #
        # local self attention
        #
        self.padding_idx=padding_idx
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap

        #
        # sampling stuff
        #
        self.sample_train_type = sample_train_type
        self.sample_n = sample_n
        self.sample_context = sample_context

        if self.sample_context == "ck":
            i = 3
            self.sample_cnn3 = nn.Sequential(
                        nn.ConstantPad1d((0,i - 1), 0),
                        nn.Conv1d(kernel_size=i, in_channels=self.bert_model.config.dim, out_channels=self.bert_model.config.dim),
                        nn.ReLU()
                        ) 
        elif self.sample_context == "ck-small":
            i = 3
            self.sample_projector = nn.Linear(self.bert_model.config.dim,384)
            self.sample_cnn3 = nn.Sequential(
                        nn.ConstantPad1d((0,i - 1), 0),
                        nn.Conv1d(kernel_size=i, in_channels=384, out_channels=128),
                        nn.ReLU()
                        ) 
        elif self.sample_context == "tk":
            self.tk_projector = nn.Linear(self.bert_model.config.dim,384)
            encoder_layer = nn.TransformerEncoderLayer(384, 8, dim_feedforward=384, dropout=0)
            self.tk_contextualizer = nn.TransformerEncoder(encoder_layer, 1, norm=None)
            self.tK_mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.sampling_binweights = nn.Linear(11, 1, bias=True)
        torch.nn.init.uniform_(self.sampling_binweights.weight, -0.01, 0.01)
        self.kernel_alpha_scaler = nn.Parameter(torch.full([1,1,11], 1, dtype=torch.float32, requires_grad=True))

        self.register_buffer("mu",nn.Parameter(torch.tensor([1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]), requires_grad=False).view(1, 1, 1, -1))
        self.register_buffer("sigma", nn.Parameter(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), requires_grad=False).view(1, 1, 1, -1))
        


    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False,
                bert_part_cached:Union[bool, torch.Tensor] = False) -> Dict[str, torch.Tensor]:

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
        orig_packed_indices = packed_indices.clone()
        ids_packed = chunked_ids_unrolled[packed_indices]
        mask_packed = (ids_packed != self.padding_idx)

        total_chunks=chunked_ids_unrolled.shape[0]

        packed_query_ids = query["input_ids"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["input_ids"].shape[1])[packed_indices]
        packed_query_mask = query["attention_mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["attention_mask"].shape[1])[packed_indices]

        #
        # sampling
        # 
        if self.sample_n > -1:
            
            #
            # id - matches
            #
            #packed_patch_scores = torch.log(1 + ((packed_query_ids.unsqueeze(-1) == ids_packed.unsqueeze(1)) * mask_packed.unsqueeze(1)).sum(-1).float()).sum(-1,keepdim=True)
            
            #
            # first-k
            #
            #sampling_scores_per_doc = torch.zeros((batch_size,chunk_pieces), device = packed_query_ids.device)
            #sampling_scores_per_doc[:,0] = 3
            #sampling_scores_per_doc[:,1] = 2
            #sampling_scores_per_doc[:,2] = 1

            #
            # ck learned matches
            #
            if self.sample_context == "ck-small":
                query_ctx = torch.nn.functional.normalize(self.sample_cnn3(self.sample_projector(self.bert_model.embeddings(packed_query_ids).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
                document_ctx = torch.nn.functional.normalize(self.sample_cnn3(self.sample_projector(self.bert_model.embeddings(ids_packed).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
            elif self.sample_context == "ck":
                query_ctx = torch.nn.functional.normalize(self.sample_cnn3((self.bert_model.embeddings(packed_query_ids).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
                document_ctx = torch.nn.functional.normalize(self.sample_cnn3((self.bert_model.embeddings(ids_packed).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
            else:
                qe = self.tk_projector(self.bert_model.embeddings(packed_query_ids).detach())
                de = self.tk_projector(self.bert_model.embeddings(ids_packed).detach())
                query_ctx = self.tk_contextualizer(qe.transpose(1,0),src_key_padding_mask=~packed_query_mask.bool()).transpose(1,0)
                document_ctx = self.tk_contextualizer(de.transpose(1,0),src_key_padding_mask=~mask_packed.bool()).transpose(1,0)
        
                query_ctx =   torch.nn.functional.normalize(query_ctx,p=2,dim=-1)
                document_ctx= torch.nn.functional.normalize(document_ctx,p=2,dim=-1)
                #query_ctx =   torch.nn.functional.normalize((self.tK_mixer * qe) + ((1 - self.tK_mixer) * query_ctx),p=2,dim=-1)
                #document_ctx= torch.nn.functional.normalize((self.tK_mixer * de) + ((1 - self.tK_mixer) * document_ctx),p=2,dim=-1)

            cosine_matrix = torch.bmm(query_ctx,document_ctx.transpose(-1, -2)).unsqueeze(-1)

            kernel_activations = torch.exp(- torch.pow(cosine_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2))) * mask_packed.unsqueeze(-1).unsqueeze(1)
            kernel_res = torch.log(torch.clamp(torch.sum(kernel_activations, 2) * self.kernel_alpha_scaler, min=1e-4)) * packed_query_mask.unsqueeze(-1)
            packed_patch_scores = self.sampling_binweights(torch.sum(kernel_res, 1))

            
            sampling_scores_per_doc = torch.zeros((total_chunks,1), dtype=packed_patch_scores.dtype, layout=packed_patch_scores.layout, device=packed_patch_scores.device)
            sampling_scores_per_doc[packed_indices] = packed_patch_scores
            sampling_scores_per_doc = sampling_scores_per_doc.reshape(batch_size,-1,)
            sampling_scores_per_doc_orig = sampling_scores_per_doc.clone()
            sampling_scores_per_doc[sampling_scores_per_doc == 0] = -9000

            sampling_sorted = sampling_scores_per_doc.sort(descending=True)
            sampled_indices = sampling_sorted.indices + torch.arange(0,sampling_scores_per_doc.shape[0]*sampling_scores_per_doc.shape[1],sampling_scores_per_doc.shape[1],device=sampling_scores_per_doc.device).unsqueeze(-1)

            sampled_indices = sampled_indices[:,:self.sample_n]
            sampled_indices_mask = torch.zeros_like(packed_indices).scatter(0, sampled_indices.reshape(-1), 1)

            if not self.training and (type(bert_part_cached) == bool and bert_part_cached == False):
                packed_indices = sampled_indices_mask * packed_indices
    
                packed_query_ids = query["input_ids"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["input_ids"].shape[1])[packed_indices]
                packed_query_mask = query["attention_mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["attention_mask"].shape[1])[packed_indices]

                ids_packed = chunked_ids_unrolled[packed_indices]
                mask_packed = (ids_packed != self.padding_idx)
                #positions_packed = chunk_positions[packed_indices]

        #
        # expensive bert scores
        #
        with torch.set_grad_enabled(self.sample_n == -1 and self.training): # this deactivates (memory)costly gradients for bert
            if self.sample_n > -1:
                self.bert_model.eval() # this deactivates dropout
        
            if bert_part_cached == None or (type(bert_part_cached) == bool and bert_part_cached == False):
                bert_vecs = self.forward_representation(torch.cat([packed_query_ids,ids_packed],dim=1),torch.cat([packed_query_mask,mask_packed],dim=1))
                packed_patch_scores = self._classification_layer(bert_vecs) 

                scores_per_doc = torch.zeros((total_chunks,1), dtype=packed_patch_scores.dtype, layout=packed_patch_scores.layout, device=packed_patch_scores.device)
                scores_per_doc[packed_indices] = packed_patch_scores
                scores_per_doc = scores_per_doc.reshape(batch_size,-1,)
                scores_per_doc_orig = scores_per_doc.clone()
                scores_per_doc_orig_sorter = scores_per_doc.clone()

            else:
                if bert_part_cached.shape[0] != batch_size or bert_part_cached.shape[1] != chunk_pieces:
                    raise Exception("cache sanity check failed! should be:"+ str(batch_size) +"," + str(chunk_pieces)+ " but is: "+str(bert_part_cached.shape[0])+"," +str(bert_part_cached.shape[1]) )
                scores_per_doc = bert_part_cached
                scores_per_doc_orig = bert_part_cached
                scores_per_doc_orig_sorter = bert_part_cached.clone()

            if self.sample_n > -1:
                scores_per_doc = scores_per_doc * sampled_indices_mask.view(batch_size,-1)
            
            #if not self.training:

            if scores_per_doc.shape[1] < self.top_k_chunks:
                scores_per_doc = nn.functional.pad(scores_per_doc,(0, self.top_k_chunks - scores_per_doc.shape[1]))

            scores_per_doc[scores_per_doc == 0] = -9000
            scores_per_doc_orig_sorter[scores_per_doc_orig_sorter == 0] = -9000
            score = torch.sort(scores_per_doc,descending=True,dim=-1).values
            score[score <= -8900] = 0

            score = (score[:,:self.top_k_chunks] * self.top_k_scoring).sum(dim=1)

        if self.sample_n == -1:
            if output_secondary_output:
                return score,{
                    "packed_indices": orig_packed_indices.view(batch_size,-1),
                    "bert_scores":scores_per_doc_orig
                }
            else:
                return score #,scores_per_doc_orig    
        else:
            if output_secondary_output:
                return score,scores_per_doc_orig,{
                    "score": score,
                    "document_ids":document_ids,
                    "packed_indices": orig_packed_indices.view(batch_size,-1),
                    "sampling_scores":sampling_scores_per_doc_orig,
                    "bert_scores":scores_per_doc_orig
                } ,None,None
            if self.sample_train_type=="mseloss":
                return score,scores_per_doc_orig,[[torch.nn.MSELoss()(sampling_scores_per_doc_orig,scores_per_doc_orig.detach())]],[sampling_sorted.indices,scores_per_doc_orig_sorter.sort(descending=True).indices]
            elif self.sample_train_type=="kldivloss":
                return score,scores_per_doc_orig,[[torch.nn.KLDivLoss(reduction="batchmean")(torch.softmax(sampling_scores_per_doc_orig,-1),torch.softmax(scores_per_doc_orig,-1).detach())]],[sampling_sorted.indices,scores_per_doc_orig_sorter.sort(descending=True).indices]
            elif self.sample_train_type=="crossentropy":
                return score,scores_per_doc_orig,[[SoftCrossEntropy()(sampling_scores_per_doc_orig,torch.softmax(scores_per_doc_orig,-1).detach())]],[sampling_sorted.indices,scores_per_doc_orig_sorter.sort(descending=True).indices]
            elif self.sample_train_type=="lambdaloss":

                bert_gains_indices = torch.sort(scores_per_doc_orig_sorter,descending=True,dim=-1).indices + torch.arange(0,scores_per_doc_orig_sorter.shape[0]*scores_per_doc_orig_sorter.shape[1],scores_per_doc_orig_sorter.shape[1],device=scores_per_doc_orig_sorter.device).unsqueeze(-1)

                bert_gains = torch.zeros_like(packed_indices).float()
                for i in range(self.sample_n):
                    bert_gains.scatter_(0, bert_gains_indices[:,i].reshape(-1), self.sample_n - i)
                bert_gains[~packed_indices] = -9000

                return score,scores_per_doc_orig,[[LambdaLoss("ndcgLoss2_scheme")(sampling_scores_per_doc,bert_gains.view(batch_size,-1).detach(),padded_value_indicator=-9000)]],[sampling_sorted.indices,scores_per_doc_orig_sorter.sort(descending=True).indices]

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
        return "IDCM: sampling.conv_binweights: " + str(self.sampling_binweights.weight.data)+ str(self.sampling_binweights.bias) +"kernel_alpha_scaler" +str(self.kernel_alpha_scaler) +\
               "top_k_scoring:" + str(self.top_k_scoring.data)

    def get_param_secondary(self):
        return {}

from transformers import PreTrainedModel,PretrainedConfig

class IDCM_Config(PretrainedConfig):
    model_type="IDCM"
    
    bert_model:str
    # how many passages get scored by BERT 
    sample_n:int

    # type of fast module
    sample_context:str

    # how many passages to take from bert to create the final score (usually the same as sample_n, but could be set to 1 for max-p)
    top_k_chunks:int

    # window size
    chunk_size:int

    # left and right overlap (added to each window)
    overlap:int 

    padding_idx:int

class IDCM_InferenceOnly(PreTrainedModel):
    '''
    IDCM is a neural re-ranking model for long documents, it creates an intra-document cascade between a fast (CK) and a slow module (BERT_Cat)
    This code is only usable for inference (we removed the training mechanism for simplicity)
    '''

    config_class = IDCM_Config
    base_model_prefix = "bert_model"

    @staticmethod
    def from_config(config, padding_idx):
        cfg = IDCM_Config()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.sample_n = config["idcm_sample_n"]
        cfg.sample_context =config["idcm_sample_context"]
        cfg.top_k_chunks = config["idcm_top_k_chunks"]
        cfg.chunk_size = config["idcm_chunk_size"]
        cfg.overlap = config["idcm_overlap"]
        cfg.padding_idx = padding_idx
        return IDCM_InferenceOnly(cfg)

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)

        #
        # bert - scoring
        #
        if isinstance(cfg.bert_model, str):
            self.bert_model = AutoModel.from_pretrained(cfg.bert_model)
        else:
            self.bert_model = cfg.bert_model

        #
        # final scoring (combination of bert scores)
        #
        self._classification_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)
        self.top_k_chunks = cfg.top_k_chunks
        self.top_k_scoring = nn.Parameter(torch.full([1,self.top_k_chunks], 1, dtype=torch.float32, requires_grad=True))

        #
        # local self attention
        #
        self.padding_idx= cfg.padding_idx
        self.chunk_size = cfg.chunk_size
        self.overlap = cfg.overlap
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap

        #
        # sampling stuff
        #
        self.sample_n = cfg.sample_n
        self.sample_context = cfg.sample_context

        if self.sample_context == "ck":
            i = 3
            self.sample_cnn3 = nn.Sequential(
                        nn.ConstantPad1d((0,i - 1), 0),
                        nn.Conv1d(kernel_size=i, in_channels=self.bert_model.config.dim, out_channels=self.bert_model.config.dim),
                        nn.ReLU()
                        ) 
        elif self.sample_context == "ck-small":
            i = 3
            self.sample_projector = nn.Linear(self.bert_model.config.dim,384)
            self.sample_cnn3 = nn.Sequential(
                        nn.ConstantPad1d((0,i - 1), 0),
                        nn.Conv1d(kernel_size=i, in_channels=384, out_channels=128),
                        nn.ReLU()
                        ) 

        self.sampling_binweights = nn.Linear(11, 1, bias=True)
        torch.nn.init.uniform_(self.sampling_binweights.weight, -0.01, 0.01)
        self.kernel_alpha_scaler = nn.Parameter(torch.full([1,1,11], 1, dtype=torch.float32, requires_grad=True))

        self.register_buffer("mu",nn.Parameter(torch.tensor([1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]), requires_grad=False).view(1, 1, 1, -1))
        self.register_buffer("sigma", nn.Parameter(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), requires_grad=False).view(1, 1, 1, -1))
        

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:

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
        orig_packed_indices = packed_indices.clone()
        ids_packed = chunked_ids_unrolled[packed_indices]
        mask_packed = (ids_packed != self.padding_idx)

        total_chunks=chunked_ids_unrolled.shape[0]

        packed_query_ids = query["input_ids"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["input_ids"].shape[1])[packed_indices]
        packed_query_mask = query["attention_mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["attention_mask"].shape[1])[packed_indices]

        #
        # sampling
        # 
        if self.sample_n > -1:
            
            #
            # id - matches
            #
            #packed_patch_scores = torch.log(1 + ((packed_query_ids.unsqueeze(-1) == ids_packed.unsqueeze(1)) * mask_packed.unsqueeze(1)).sum(-1).float()).sum(-1,keepdim=True)
            
            #
            # first-k
            #
            #sampling_scores_per_doc = torch.zeros((batch_size,chunk_pieces), device = packed_query_ids.device)
            #sampling_scores_per_doc[:,0] = 3
            #sampling_scores_per_doc[:,1] = 2
            #sampling_scores_per_doc[:,2] = 1

            #
            # ck learned matches
            #
            if self.sample_context == "ck-small":
                query_ctx = torch.nn.functional.normalize(self.sample_cnn3(self.sample_projector(self.bert_model.embeddings(packed_query_ids).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
                document_ctx = torch.nn.functional.normalize(self.sample_cnn3(self.sample_projector(self.bert_model.embeddings(ids_packed).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
            elif self.sample_context == "ck":
                query_ctx = torch.nn.functional.normalize(self.sample_cnn3((self.bert_model.embeddings(packed_query_ids).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
                document_ctx = torch.nn.functional.normalize(self.sample_cnn3((self.bert_model.embeddings(ids_packed).detach()).transpose(1,2)).transpose(1, 2),p=2,dim=-1)
            else:
                qe = self.tk_projector(self.bert_model.embeddings(packed_query_ids).detach())
                de = self.tk_projector(self.bert_model.embeddings(ids_packed).detach())
                query_ctx = self.tk_contextualizer(qe.transpose(1,0),src_key_padding_mask=~packed_query_mask.bool()).transpose(1,0)
                document_ctx = self.tk_contextualizer(de.transpose(1,0),src_key_padding_mask=~mask_packed.bool()).transpose(1,0)
        
                query_ctx =   torch.nn.functional.normalize(query_ctx,p=2,dim=-1)
                document_ctx= torch.nn.functional.normalize(document_ctx,p=2,dim=-1)

            cosine_matrix = torch.bmm(query_ctx,document_ctx.transpose(-1, -2)).unsqueeze(-1)

            kernel_activations = torch.exp(- torch.pow(cosine_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2))) * mask_packed.unsqueeze(-1).unsqueeze(1)
            kernel_res = torch.log(torch.clamp(torch.sum(kernel_activations, 2) * self.kernel_alpha_scaler, min=1e-4)) * packed_query_mask.unsqueeze(-1)
            packed_patch_scores = self.sampling_binweights(torch.sum(kernel_res, 1))

            
            sampling_scores_per_doc = torch.zeros((total_chunks,1), dtype=packed_patch_scores.dtype, layout=packed_patch_scores.layout, device=packed_patch_scores.device)
            sampling_scores_per_doc[packed_indices] = packed_patch_scores
            sampling_scores_per_doc = sampling_scores_per_doc.reshape(batch_size,-1,)
            sampling_scores_per_doc_orig = sampling_scores_per_doc.clone()
            sampling_scores_per_doc[sampling_scores_per_doc == 0] = -9000

            sampling_sorted = sampling_scores_per_doc.sort(descending=True)
            sampled_indices = sampling_sorted.indices + torch.arange(0,sampling_scores_per_doc.shape[0]*sampling_scores_per_doc.shape[1],sampling_scores_per_doc.shape[1],device=sampling_scores_per_doc.device).unsqueeze(-1)

            sampled_indices = sampled_indices[:,:self.sample_n]
            sampled_indices_mask = torch.zeros_like(packed_indices).scatter(0, sampled_indices.reshape(-1), 1)

            # pack indices

            packed_indices = sampled_indices_mask * packed_indices
    
            packed_query_ids = query["input_ids"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["input_ids"].shape[1])[packed_indices]
            packed_query_mask = query["attention_mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query["attention_mask"].shape[1])[packed_indices]

            ids_packed = chunked_ids_unrolled[packed_indices]
            mask_packed = (ids_packed != self.padding_idx)

        #
        # expensive bert scores
        #
        
        bert_vecs = self.forward_representation(torch.cat([packed_query_ids,ids_packed],dim=1),torch.cat([packed_query_mask,mask_packed],dim=1))
        packed_patch_scores = self._classification_layer(bert_vecs) 

        scores_per_doc = torch.zeros((total_chunks,1), dtype=packed_patch_scores.dtype, layout=packed_patch_scores.layout, device=packed_patch_scores.device)
        scores_per_doc[packed_indices] = packed_patch_scores
        scores_per_doc = scores_per_doc.reshape(batch_size,-1,)
        scores_per_doc_orig = scores_per_doc.clone()
        scores_per_doc_orig_sorter = scores_per_doc.clone()

        if self.sample_n > -1:
            scores_per_doc = scores_per_doc * sampled_indices_mask.view(batch_size,-1)
        
        #
        # aggregate bert scores
        #

        if scores_per_doc.shape[1] < self.top_k_chunks:
            scores_per_doc = nn.functional.pad(scores_per_doc,(0, self.top_k_chunks - scores_per_doc.shape[1]))

        scores_per_doc[scores_per_doc == 0] = -9000
        scores_per_doc_orig_sorter[scores_per_doc_orig_sorter == 0] = -9000
        score = torch.sort(scores_per_doc,descending=True,dim=-1).values
        score[score <= -8900] = 0

        score = (score[:,:self.top_k_chunks] * self.top_k_scoring).sum(dim=1)

        if self.sample_n == -1:
            if output_secondary_output:
                return score,{
                    "packed_indices": orig_packed_indices.view(batch_size,-1),
                    "bert_scores":scores_per_doc_orig
                }
            else:
                return score,scores_per_doc_orig    
        else:
            if output_secondary_output:
                return score,scores_per_doc_orig,{
                    "score": score,
                    "packed_indices": orig_packed_indices.view(batch_size,-1),
                    "sampling_scores":sampling_scores_per_doc_orig,
                    "bert_scores":scores_per_doc_orig
                }

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
        return "IDCM: sampling.conv_binweights: " + str(self.sampling_binweights.weight.data)+ str(self.sampling_binweights.bias) +"kernel_alpha_scaler" +str(self.kernel_alpha_scaler) +\
               "top_k_scoring:" + str(self.top_k_scoring.data)

    def get_param_secondary(self):
        return {}
