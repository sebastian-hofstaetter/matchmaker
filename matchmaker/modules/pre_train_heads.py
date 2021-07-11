from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation


class PreTrain_MLM_Head(nn.Module):
    '''
    pre-train masked language model head (wiht 1 layer classifier)
    '''

    def __init__(self,
                 neural_ir_model: nn.Module,
                 representation_size:int,
                 vocab_size:int):

        super(PreTrain_MLM_Head, self).__init__()

        self.neural_ir_model = neural_ir_model
        self.mlm_head = nn.Linear(representation_size, vocab_size)
        self.loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear): # broken !!
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq: Dict[str, torch.Tensor],target: torch.Tensor,accuracy_type2: torch.Tensor,use_fp16,title:torch.Tensor=None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=use_fp16):
            if title is None:
                vectors = self.neural_ir_model.forward_representation(seq, sequence_type="pretrain")
            else:
                vectors = self.neural_ir_model.forward_representation(seq, title, sequence_type="pretrain")

            pred_scores = self.mlm_head(vectors)

            loss = self.loss_fct(pred_scores.view(-1, pred_scores.shape[-1]), target.view(-1))

            perplexity = torch.exp(loss.detach())

            _, predicted = torch.max(pred_scores.view(-1, pred_scores.shape[-1]),1)
            correct = (predicted == target.view(-1)).sum().float()
            sum_target = (target > -100).sum().float()
            accuracy = 100 * correct / sum_target

            target2 = target.clone()
            target2[~accuracy_type2.bool()] = - 100
            sum_target2 = (target2 > -100).sum().float()
            correct2 = (predicted == target2.view(-1)).sum().float()
            if sum_target2 > 0:
                accuracy2 = 100 * correct2 / sum_target2
            else:
                accuracy2 = correct2

            return loss,perplexity,accuracy,accuracy2

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()

class PreTrain_MLM_POD_Head(nn.Module):
    '''
    pre-train masked language model head (wiht 1 layer classifier)
    '''

    def __init__(self,
                 neural_ir_model: nn.Module,
                 representation_size:int,
                 vocab_size:int):

        super(PreTrain_MLM_POD_Head, self).__init__()

        self.neural_ir_model = neural_ir_model
        self.mlm_head = nn.Linear(representation_size, vocab_size)
        self.loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        self.loss_fct2 = torch.nn.BCEWithLogitsLoss(reduction="none") 
        self.loss_fct3 = torch.nn.BCEWithLogitsLoss(reduction="mean") 
        #self.loss_fct3 = torch.nn.CosineEmbeddingLoss() 
        #self.loss_fct2 = torch.nn.BCELoss(reduction="none") 

        self.apply(self._init_weights)
        self.check_inter_passage_pod=False
        self.doc_pass_pod=True

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear): # broken !!
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            torch.nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq: Dict[str, torch.Tensor],target: torch.Tensor,accuracy_type2: torch.Tensor,use_fp16,title:torch.Tensor=None,title_target:torch.Tensor=None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=use_fp16):
            if title is None:
                vectors,topic_vecs,title_vecs,part_vecs,part_mask = self.neural_ir_model.forward_representation(seq, sequence_type="pretrain")
            else:
                vectors,topic_vecs,title_vecs,part_vecs,part_mask = self.neural_ir_model.forward_representation(seq, title, sequence_type="pretrain")
            
            title_pred = self.mlm_head(title_vecs[:,1:,:])
            loss_title = self.loss_fct(title_pred.view(-1, title_pred.shape[-1]), title_target[:,1:].reshape(-1))

            #
            # mlm over all vecs
            #
            pred_scores = self.mlm_head(vectors)

            loss = self.loss_fct(pred_scores.view(-1, pred_scores.shape[-1]), target.view(-1))
            perplexity = torch.exp(loss.detach())

            _, predicted = torch.max(pred_scores.view(-1, pred_scores.shape[-1]),1)
            correct = (predicted == target.view(-1)).sum().float()
            sum_target = (target > -100).sum().float()
            accuracy = 100 * correct / sum_target

            target2 = target.clone()
            target2[~accuracy_type2.bool()] = - 100
            sum_target2 = (target2 > -100).sum().float()
            correct2 = (predicted == target2.view(-1)).sum().float()
            if sum_target2 > 0:
                accuracy2 = 100 * correct2 / sum_target2
            else:
                accuracy2 = correct2

            #
            # pod - part of document loss
            #
            if self.doc_pass_pod:
                doc_vecs = topic_vecs[:,0,:]
                is_in_doc_target = torch.zeros((part_vecs.shape[0],part_vecs.shape[0]),device=part_vecs.device)
                is_in_doc_target.fill_diagonal_(1)
                is_in_doc_target_pod = is_in_doc_target.repeat_interleave(part_vecs.shape[1],dim=1)

                #doc_vecs = doc_vecs / (doc_vecs.norm(p=2,dim=1, keepdim=True) + 0.0001)
                #part_vecs = part_vecs.view(-1,part_vecs.shape[-1])
                #part_vecs = part_vecs / (part_vecs.norm(p=2,dim=-1, keepdim=True)+ 0.0001)

                if self.check_inter_passage_pod:
                    part_vecs_pod = part_vecs[part_mask]
                    doc_parts_pod = torch.cat([doc_vecs,part_vecs_pod.view(-1,part_vecs_pod.shape[-1])],dim=0)            
                    is_in_doc_target_pod = torch.cat([is_in_doc_target_pod,is_in_doc_target_pod.repeat_interleave(part_vecs.shape[1],dim=0)[part_mask.view(-1)]],dim=0)
                else:
                    doc_parts_pod = doc_vecs
                
                scaling = float(doc_parts_pod.shape[-1]) ** -0.5

                is_in_doc = torch.mm(doc_parts_pod * scaling,part_vecs.view(-1,part_vecs.shape[-1]).transpose(0, 1))

                inter_pod_diff = torch.mm(part_vecs.view(-1,part_vecs.shape[-1])[part_mask.view(-1)] * scaling, part_vecs.view(-1,part_vecs.shape[-1])[part_mask.view(-1)].transpose(0, 1))

                inter_pod_diff_target = torch.ones_like(inter_pod_diff)
                inter_pod_diff_target.fill_diagonal_(0)

                inter_pod_diff = torch.sigmoid(inter_pod_diff[inter_pod_diff_target.bool()] / 100)
                inter_pod_diff = inter_pod_diff.mean()

                #in_doc_loss = self.loss_fct2(is_in_doc.view(-1),is_in_doc_target.view(-1))
                in_doc_loss = self.loss_fct2(is_in_doc,is_in_doc_target_pod) 

                neg_cosine_margin = 0

                in_doc_loss = in_doc_loss[:,part_mask.view(-1)]

                pod_pos = is_in_doc[:,part_mask.view(-1)].view(-1)[is_in_doc_target_pod[:,part_mask.view(-1)].view(-1) == 1]#.mean()
                pod_neg = is_in_doc[:,part_mask.view(-1)].view(-1)[is_in_doc_target_pod[:,part_mask.view(-1)].view(-1) == 0]#.mean()

                #in_doc_loss = (1 - torch.tanh(pod_pos)).mean() + torch.clamp(torch.tanh(pod_neg+neg_cosine_margin), min=0).mean()
                in_doc_loss = in_doc_loss.mean() + inter_pod_diff

                pod_pos_mean = pod_pos.mean()
                pod_neg_mean = pod_neg.mean()

                #
                # topic vs. title vec
                #
                title_vecs = title_vecs[:,0,:]
                #title_vecs = title_vecs / (title_vecs.norm(p=2,dim=1, keepdim=True) + 0.0001)

                title_topic_cosine = torch.mm(doc_vecs * scaling,title_vecs.transpose(-2, -1))

                title_pos = title_topic_cosine[is_in_doc_target == 1]
                title_neg = title_topic_cosine[is_in_doc_target == 0]

                #title_topic_loss = (1 - torch.tanh(title_pos)).mean() + torch.clamp(torch.tanh(title_neg+neg_cosine_margin), min=0).mean()
                title_topic_loss = self.loss_fct3(title_topic_cosine,is_in_doc_target)

                title_pos_mean = title_pos.mean()
                title_neg_mean = title_neg.mean()

            else:
                in_doc_loss=None
                pod_pos_mean=None
                pod_neg_mean=None
                title_topic_cosine=None
            return loss,perplexity,accuracy,accuracy2,loss_title,in_doc_loss,pod_pos_mean,pod_neg_mean,title_topic_loss,title_pos_mean,title_neg_mean,inter_pod_diff

    def get_param_stats(self):
        return self.neural_ir_model.get_param_stats()

    def get_param_secondary(self):
        return self.neural_ir_model.get_param_secondary()
