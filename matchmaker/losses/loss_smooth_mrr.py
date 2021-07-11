import torch.nn as nn
import torch

class SmoothRank(nn.Module):

    def __init__(self):
        super(SmoothRank, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, scores):
        x_0 = scores.unsqueeze(dim=-1)                                              # [Q x D] --> [Q x D x 1]
        x_1 = scores.unsqueeze(dim=-2)                                              # [Q x D] --> [Q x 1 x D]
        diff = x_1 - x_0                                                            # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        is_lower = self.sigmoid(diff)                                               # [Q x D x D] --> [Q x D x D]
        ranks = torch.sum(is_lower, dim=-1) + 0.5                                   # [Q x D x D] --> [Q x D]
        return ranks

class SmoothMRRLoss(nn.Module):

    def __init__(self):
        super(SmoothMRRLoss, self).__init__()
        self.soft_ranker = SmoothRank()
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, labels, agg=True):
        ranks = self.soft_ranker(scores)                                            # [Q x D] --> [Q x D]
        labels = torch.where(labels > 0, self.one, self.zero)                       # [Q x D] --> [Q x D]
        rr = labels / ranks                                                         # [Q x D], [Q x D] --> [Q x D]
        rr_max, _ = rr.max(dim=-1)                                                  # [Q x D] --> [Q]
        loss = 1 - rr_max                                                           # [Q] --> [Q]
        if agg:
            loss = loss.mean()                                                      # [Q] --> [1]
        return loss