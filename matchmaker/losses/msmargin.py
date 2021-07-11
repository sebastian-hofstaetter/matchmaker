import torch.nn as nn
import torch

class MSMarginLoss(nn.Module):
    def __init__(self):
        super(MSMarginLoss, self).__init__()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
        All inputs should be tensors of equal size
        """     
        loss = torch.mean(torch.pow((scores_pos - scores_neg) - (label_pos - label_neg),2))
        return loss

class MarginMSE_InterPassageLoss(nn.Module):
    def __init__(self):
        super(MarginMSE_InterPassageLoss, self).__init__()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
        All inputs should be tensors of equal size
        """     
        loss = torch.mean(torch.pow((scores_pos.unsqueeze(1) - scores_neg.unsqueeze(1).transpose(-1,-2)) - (label_pos[:,:scores_pos.shape[1]].unsqueeze(1) - label_neg[:,:scores_neg.shape[1]].unsqueeze(1).transpose(-1,-2)),2))
        return loss