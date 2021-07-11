from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class Duet(nn.Module):
    '''
    Paper: An Updated Duet Model for Passage Re-ranking, Mitra et al.

    Reference code : https://github.com/dfcf93/MSMARCO/blob/master/Ranking/Baselines/Duet.ipynb
    '''
    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Duet(word_embeddings_out_dim)

    def __init__(self,
                 word_embeddings_out_dim: int):

        super(Duet, self).__init__()


        NUM_HIDDEN_NODES = word_embeddings_out_dim
        POOLING_KERNEL_WIDTH_QUERY = 18
        POOLING_KERNEL_WIDTH_DOC = 100
        DROPOUT_RATE = 0

        NUM_POOLING_WINDOWS_DOC = 99
        MAX_DOC_TERMS = 2000
        MAX_QUERY_TERMS = 30
        self.cosine_module = CosineMatrixAttention()

        self.duet_local             = nn.Sequential(nn.Conv1d(MAX_DOC_TERMS, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * MAX_QUERY_TERMS, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_dist_q            = nn.Sequential(nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(POOLING_KERNEL_WIDTH_QUERY),
                                        Flatten(),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU()
                                        )
        self.duet_dist_d            = nn.Sequential(nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(POOLING_KERNEL_WIDTH_DOC, stride=1),
                                        nn.Conv1d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, kernel_size=1),
                                        nn.ReLU()
                                        )
        self.duet_dist              = nn.Sequential(Flatten(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES * NUM_POOLING_WINDOWS_DOC, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE))
        self.duet_comb              = nn.Sequential(nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES),
                                        nn.ReLU(),
                                        nn.Dropout(p=DROPOUT_RATE),
                                        nn.Linear(NUM_HIDDEN_NODES, 1),
                                        nn.ReLU())
        #self.scale                  = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight,0,0.01)
        self.duet_comb.apply(init_normal)


    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_idfs: torch.Tensor, document_idfs: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1) #*10
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1) #*10

        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        #cosine_matrix = (cosine_matrix.detach() == 1).float() # in the paper it is only exact match - why is the question ?
        # confirmed: with the above line uncommented mrr@10 = 0.20 -> now with cosine match = 0.26
        local_weighted = cosine_matrix * query_idfs

        #if ARCH_TYPE != 1:
        h_local                 = self.duet_local(local_weighted.transpose(1,2))
        #if ARCH_TYPE > 0:
        h_dist_q                = self.duet_dist_q((query_embeddings).permute(0, 2, 1))
        h_dist_d                = self.duet_dist_d((document_embeddings).permute(0, 2, 1))

        h_dist                  = self.duet_dist(h_dist_q.unsqueeze(-1)*h_dist_d)

        #y_score                     = self.duet_comb((h_local + h_dist) if ARCH_TYPE == 2 else (h_dist if ARCH_TYPE == 1 else h_local)) * self.scale
        y_score                 = self.duet_comb(h_local + h_dist) * 0.1  # * self.scale
        
        return y_score

    def get_param_stats(self):
        return "DUET: / "
