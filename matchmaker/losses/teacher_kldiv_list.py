import torch.nn as nn
import torch

class KLDivTeacherList(nn.Module):
    def __init__(self):
        super(KLDivTeacherList, self).__init__()

        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, scores, labels):
        """
        """                      
        loss = self.kl(scores.softmax(-1),labels.softmax(-1))
        return loss