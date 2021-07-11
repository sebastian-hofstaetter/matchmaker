import torch.nn as nn
import torch.nn.functional as F
import torch

class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()


    def forward(self, y_pred, y_true, eps=1e-6):
       """
       ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
       :param y_pred: predictions from the model, shape [batch_size, slate_length]
       :param y_true: ground truth labels, shape [batch_size, slate_length]
       :param eps: epsilon value, used for numerical stability
       :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
       :return: loss value, a torch.Tensor
       from: https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
       """
       #y_pred = y_pred.clone()
       #y_true = y_true.clone()

       #mask = y_true == padded_value_indicator
       #y_pred[mask] = float('-inf')
       #y_true[mask] = float('-inf')

       preds_smax = F.softmax(y_pred, dim=1)
       true_smax = F.softmax(y_true, dim=1)

       preds_smax = preds_smax + eps
       preds_log = torch.log(preds_smax)

       return torch.mean(-torch.sum(true_smax * preds_log, dim=1))