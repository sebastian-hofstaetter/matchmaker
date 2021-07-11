import torch.nn as nn
import torch

class MSERanknetTeacher(nn.Module):
    def __init__(self):
        super(MSERanknetTeacher, self).__init__()

        self.mse = torch.nn.MSELoss()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """          
        x = scores_pos - scores_neg
        loss = self.mse(scores_pos,label_pos) + self.mse(scores_neg,label_neg)
        return loss / 2 + torch.nn.BCEWithLogitsLoss()(x, torch.ones_like(x))