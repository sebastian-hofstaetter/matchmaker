import torch.nn as nn
import torch

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, scores_pos, scores_neg, probs):
        """
        Ranknet loss: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        """
                                               
        x = scores_pos - scores_neg       
        loss = self.bce(x, probs)                                         
        return loss