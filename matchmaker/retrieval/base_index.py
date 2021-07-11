from typing import Dict, Iterator, List
import numpy

class BaseNNIndexer():
    '''
    Base class for our nearest neighbor indexing operations, atm we mainly abstrcat faiss, but it should allow us to swap in other libs fairly easy
    '''

    def __init__(self, config):
        super(BaseNNIndexer, self).__init__()

        self.token_dim = config["token_dim"]
        self.use_gpu = config["faiss_use_gpu"]
        self.use_fp16 = config["token_dtype"] == "float16"

    def prepare(self, data_chunks:List[numpy.ndarray], subsample=-1):
        '''
        Train an index with (all) or only some vectors, if subsample is set to a value between 0 and 1
        '''
        pass

    def index(self, ids:List[numpy.ndarray], data_chunks:List[numpy.ndarray]):
        '''
        ids: need to be int64
        '''
        pass

    def search(self, query_vec:numpy.ndarray, top_n:int):
        '''
        query_vec: can be 2d (batch search) or 1d (single search) 
        '''
        pass
