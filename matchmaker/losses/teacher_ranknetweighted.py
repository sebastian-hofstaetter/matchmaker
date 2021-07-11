import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss


class RankNetTeacher(nn.Module):
    def __init__(self):
        super(RankNetTeacher, self).__init__()

        self.bce = torch.nn.BCEWithLogitsLoss()


    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """                      
        x = scores_pos - scores_neg
        y = label_pos - label_neg
        loss = torch.nn.BCEWithLogitsLoss(weight=y)(x, torch.ones_like(x))
        return loss