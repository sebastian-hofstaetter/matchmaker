from torch._C import device
import torch.nn as nn
import torch
from matchmaker.losses.soft_crossentropy import SoftCrossEntropy

class QA_StartEndCrossEntropy(nn.Module):
    def __init__(self):
        super(QA_StartEndCrossEntropy, self).__init__()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        #self.loss_fct_answer = nn.BCEWithLogitsLoss()

    def forward(self, qa_logits_start,qa_logits_end,labels_start,labels_end,answerability_logits=None,labels_answerability=None):
        """
        
        """
        total_loss = None
        answer_loss = None
        if qa_logits_start != None:
            start_loss = []
            end_loss = []
            for i in range(len(labels_start[0])):
                start_loss.append(self.loss_fct(qa_logits_start, labels_start[:,i]).unsqueeze(-1))
                end_loss.append(self.loss_fct(qa_logits_end[:,i], labels_end[:,i]).unsqueeze(-1))

            start_loss = torch.cat(start_loss,dim=-1).mean(dim=-1)
            end_loss = torch.cat(end_loss,dim=-1).mean(dim=-1)
            total_loss = (start_loss + end_loss) / 2

        if answerability_logits != None:
            if total_loss == None: total_loss = torch.zeros(1,device=answerability_logits.device)

            answer_loss = self.loss_fct(answerability_logits, labels_answerability)
            #total_loss = total_loss + answer_loss  * 0.5

        return total_loss.squeeze(),answer_loss.squeeze()