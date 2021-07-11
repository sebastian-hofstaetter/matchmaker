from allennlp.data.tokenizers import Token
from blingfire import *
from typing import List
import os
import numpy
import pickle
import glob
import torch

class CrossExperimentReplayCache():
    """
    basic replay (same input order) cache for arbitrary tensors, based on numpy memory mapped files
    """

    def __init__(self,path,tensor_type="float16",is_readonly=True) -> None:
        self.path = path
        self.tensor_type = tensor_type
        self.block_size = 20_000_000
        self.is_readonly = is_readonly
        self.memmap_readmode = "c" if is_readonly else "r+" # c does not save changes, but keeps them in memory otherwise pytorch complaints on startup

        self.index = [] # (storage_idx,map_start,map_end,batch_size)
        self.storage = []

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            if os.path.exists(os.path.join(self.path,"index.pickle")):
                self.index = pickle.load( open( os.path.join(self.path,"index.pickle"), "rb" ))

            for sb in sorted(glob.glob(os.path.join(self.path,"storage-block*.np")))[:-1]:
                self.storage.append(np.memmap(sb, dtype=tensor_type, mode=self.memmap_readmode, shape=(self.block_size)))
        
        if len(self.storage) == 0:
            self.storage.append(np.memmap(os.path.join(self.path,"storage-block-0000.np"), dtype=tensor_type, mode=self.memmap_readmode, shape=(self.block_size)))

        self.replay_index = 0

    def save(self,is_final=False):
        if not self.is_readonly:
            pickle.dump(self.index, open( os.path.join(self.path,"index.pickle"), "wb" ))
            if is_final:
                np.memmap(os.path.join(self.path,"storage-block-"+"{:04d}".format(len(self.storage))+".np"), 
                          dtype=self.tensor_type, mode=self.memmap_readmode, shape=(1))

    def get_next(self):

        if self.replay_index >= len(self.index):
            self.replay_index += 1
            return None

        curr = self.index[self.replay_index]

        tensor = torch.from_numpy(self.storage[curr[0]][curr[1]:curr[2]]).view(curr[3],-1)
        self.replay_index += 1

        return tensor

    def cache(self, tensor:torch.Tensor) -> None:
        if tensor == None:
            print("trying to cache empty tensor")
            return

        if self.is_readonly:
            return

        if len(self.index) == 0:
            current_storage=0
            last_end=0
        else:
            current_storage = self.index[-1][0]
            last_end = self.index[-1][2]

        current_batch_size = tensor.shape[0]
        flat_tensor = tensor.view(-1)
        current_full_size = flat_tensor.shape[0]

        if current_full_size > self.block_size:
            raise Exception("can't cache too big tensor")

        if last_end + current_full_size > self.block_size:
            self.save()
            self.storage.append(np.memmap(os.path.join(self.path,"storage-block-"+"{:04d}".format(len(self.storage))+".np"), 
                                          dtype=self.tensor_type, mode=self.memmap_readmode, shape=(self.block_size)))
            last_end = 0
            current_storage+=1

        self.storage[current_storage][last_end:last_end+current_full_size] = flat_tensor.cpu().numpy()
        self.index.append((current_storage,last_end,last_end+current_full_size,current_batch_size))
