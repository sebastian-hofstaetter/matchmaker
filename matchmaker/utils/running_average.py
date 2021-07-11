import torch

class RunningAverage():
    """
    module to hold a running average list as tensor on a gpu  
    """

    def __init__(self, size):
        super().__init__()
        self.entries = torch.zeros(size).cuda()
        self.current_entry_idx = 0

    def add_entry(self, value):
        if self.current_entry_idx == self.entries.shape[0]:
            self.current_entry_idx = 0
        
        self.entries[self.current_entry_idx] = value.detach()
        self.current_entry_idx += 1

    def get_average(self):
        return self.entries.mean()