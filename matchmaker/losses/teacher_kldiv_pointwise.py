import torch.nn as nn
import torch

class KLDivTeacherPointwise(nn.Module):
    def __init__(self):
        super(KLDivTeacherPointwise, self).__init__()

        self.kl = torch.nn.KLDivLoss()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """                      
        loss = self.kl(scores_pos,label_pos) + self.kl(scores_neg,label_neg)
        return loss / 2