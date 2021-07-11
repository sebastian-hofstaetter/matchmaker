from math import sqrt
from typing import Dict, Iterator, List
import numpy
from .base_index import BaseNNIndexer
import scann

from rich.console import Console
console = Console()

class ScaNNIndexer(BaseNNIndexer):
    '''
    ScaNN index wrapper https://github.com/google-research/google-research/tree/master/scann
    '''

    def __init__(self, config):
        super(ScaNNIndexer, self).__init__(config)

        self.token_dim = config["token_dim"]
        self.use_gpu = False
        self.use_fp16 = config["token_dtype"] == "float16"
        c = next(iter(config["query_sets"].values()))          # get the first query_set for top_n 
        self.top_n = c.get("index_hit_top_n",c["top_n"]) # need to get top_n also for index building here, not super clean but meh ...

    def index(self, ids:List[numpy.ndarray], data_chunks:List[numpy.ndarray]):
        '''
        ids: need to be int64
        '''
        self.id_mapping = numpy.concatenate(ids)#.astype(numpy.int64)
        c = numpy.concatenate(data_chunks)#.astype(numpy.float32)
        console.log("[ScaNNIndexer]","Index",len(c),"vectors; with ",int(sqrt(len(c))),"leaves")

        self.scann_index = scann.scann_ops_pybind.builder(c, self.top_n, "dot_product")\
                                                 .tree(num_leaves=int(sqrt(len(c))), num_leaves_to_search=100, training_sample_size=len(c))\
                                                 .score_ah(2, anisotropic_quantization_threshold=0.2)\
                                                 .reorder(self.top_n).build()

    def search(self, query_vec:numpy.ndarray, top_n:int):
        '''
        query_vec: can be 2d (batch search) or 1d (single search) 
        '''
        if len(query_vec.shape) == 1:
            query_vec = query_vec[numpy.newaxis,:]

        neighbors, distances = self.scann_index.search_batched(query_vec, final_num_neighbors=top_n)

        neighbor_ids = self.id_mapping[neighbors]
        return (distances,neighbor_ids)

    def save(self, path:str):
        self.scann_index.serialize(path)
    
    def load(self, path:str):
        self.scann_index = scann.scann_ops_pybind.load_searcher(path)
