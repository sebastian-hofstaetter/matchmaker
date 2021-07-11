import torch.nn as nn
import torch

class MSETeacherPointwise(nn.Module):
    def __init__(self):
        super(MSETeacherPointwise, self).__init__()

        self.mse = torch.nn.MSELoss()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """                      
        loss = self.mse(scores_pos,label_pos) + self.mse(scores_neg,label_neg)
        return loss / 2

class MSETeacherPointwisePassages(nn.Module):
    def __init__(self):
        super(MSETeacherPointwisePassages, self).__init__()

        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """
        #label_pos[label_pos != 0] += 115
        #label_neg[label_neg != 0] += 115

        loss = self.mse(scores_pos,label_pos[:,:scores_pos.shape[1]].detach())[label_pos[:,:scores_pos.shape[1]] != 0].mean() \
             + self.mse(scores_neg,label_neg[:,:scores_neg.shape[1]].detach())[label_neg[:,:scores_neg.shape[1]] != 0].mean()
        return loss / 2