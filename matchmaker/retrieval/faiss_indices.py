from typing import Dict, Iterator, List

import faiss
import numpy
import os
import gc
#from faiss.contrib.ondisk import merge_ondisk
from rich.console import Console
console = Console()
from .base_index import BaseNNIndexer


class FaissBaseIndexer(BaseNNIndexer):
    '''
    Shared faiss code
    '''

    def __init__(self,config):
        super(FaissBaseIndexer, self).__init__(config)
        self.faiss_index:faiss.Index = None # needs to be initialized by the actual faiss classes

    def index(self, ids:List[numpy.ndarray], data_chunks:List[numpy.ndarray]):
        # single add needed for multi-gpu index (sharded), and hnsw so just do it for all (might be a memory problem at some point, but we can come back to that)
        i = numpy.concatenate(ids).astype(numpy.int64)
        c = numpy.concatenate(data_chunks).astype(numpy.float32)
        console.log("[FaissIndexer]","Add",c.shape[0]," vectors")
        self.faiss_index.add_with_ids(c,i)

    def search(self, query_vec:numpy.ndarray, top_n:int):
        # even a single search must be 1xn dims
        if len(query_vec.shape) == 1:
            query_vec = query_vec[numpy.newaxis,:]
            
        res_scores, indices = self.faiss_index.search(query_vec.astype(numpy.float32),top_n)

        return res_scores, indices

    def save(self, path:str):
        if self.use_gpu:
            idx = faiss.index_gpu_to_cpu(self.faiss_index)
        else:
            idx = self.faiss_index
        faiss.write_index(idx, path)
    
    def load(self, path:str,config_overwrites=None):
        self.faiss_index = faiss.read_index(path)


class FaissIdIndexer(FaissBaseIndexer):
    '''
    Simple brute force nearest neighbor faiss index with id mappings, with potential gpu usage, support for fp16
    -> if faiss_use_gpu=True use all availbale GPUs in a sharded index 
    '''

    def __init__(self,config):
        super(FaissIdIndexer, self).__init__(config)

        if self.use_gpu:

            console.log("[FaissIdIndexer]","Index on GPU")
            cpu_index = faiss.IndexIDMap(faiss.IndexFlatIP(config["token_dim"]))
                        
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = self.use_fp16

            self.faiss_index = faiss.index_cpu_to_all_gpus(cpu_index,co)

        else:
            console.log("[FaissIdIndexer]","Index on CPU")
            if self.use_fp16:
                self.faiss_index = faiss.IndexIDMap(faiss.IndexScalarQuantizer(config["token_dim"],faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT))
            else:
                self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(config["token_dim"]))

class FaissHNSWIndexer(FaissBaseIndexer):
    '''
    HNSW - graph based - index, only supports CPU - but gets very low query latency 
    '''

    def __init__(self,config):
        super(FaissHNSWIndexer, self).__init__(config)

        self.use_gpu = False # HNSW does not support GPUs

        if self.use_fp16:
            console.log("[FaissHNSWIndexer]","Index with fp16")
            self.faiss_index = faiss.IndexHNSWSQ(config["token_dim"],faiss.ScalarQuantizer.QT_fp16,
                                                config["faiss_hnsw_graph_neighbors"],faiss.METRIC_INNER_PRODUCT)

        else:
            console.log("[FaissHNSWIndexer]","Index with fp32")
            self.faiss_index = faiss.IndexHNSWFlat(config["token_dim"],config["faiss_hnsw_graph_neighbors"],faiss.METRIC_INNER_PRODUCT)
        
        self.faiss_index.verbose = True
        self.faiss_index.hnsw.efConstruction = config["faiss_hnsw_efConstruction"]
        self.faiss_index.hnsw.efSearch = config["faiss_hnsw_efSearch"]

        self.faiss_index = faiss.IndexIDMap(self.faiss_index)

    def prepare(self, data_chunks:List[numpy.ndarray], subsample=-1):
        if self.use_fp16:
            # training for the scalar quantizer, according to: https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py
            self.faiss_index.train(numpy.concatenate(data_chunks).astype(numpy.float32))

class FaissIVFIndexer(FaissBaseIndexer):
    '''
    IVF indexer (creates clusters on train)
    '''

    def __init__(self, config):
        super(FaissIVFIndexer, self).__init__(config)

        self.list_n_probe = config["faiss_ivf_search_probe_count"]
        self.faiss_ivf_list_count = config["faiss_ivf_list_count"]

        if self.use_fp16:
            console.log("[FaissIVFIndexer]","Use FP16")
            self.faiss_index = faiss.IndexIVFScalarQuantizer(faiss.IndexScalarQuantizer(self.token_dim, faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT),
                                                             self.token_dim, self.faiss_ivf_list_count, faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
        else:
            console.log("[FaissIVFIndexer]","Use FP32")
            self.faiss_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.token_dim),self.token_dim,self.faiss_ivf_list_count,faiss.METRIC_INNER_PRODUCT)

        self.faiss_index.nprobe = self.list_n_probe

        if self.use_gpu:
            console.log("[FaissIVFIndexer]","Use GPU")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = self.use_fp16

            self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index,co)

    def prepare(self, data_chunks:List[numpy.ndarray], subsample=-1):
        '''
        subsample = -1, train on all vecs, if 0 - 1 train on fraction rand subset
        '''
        all_vecs = numpy.concatenate(data_chunks).astype(numpy.float32)
        console.log("[FaissIVFIndexer]","Train",len(all_vecs),"entries",self.faiss_ivf_list_count,"lists (avg-goal/list: ",len(all_vecs)//self.faiss_ivf_list_count,")")
        self.faiss_index.train(all_vecs)

    def load(self, path:str, config_overwrites=None):
        super().load(path)
        self.faiss_index.nprobe = config_overwrites["faiss_ivf_search_probe_count"]

def merge_ondisk(trained_index: faiss.Index,
                 shard_fnames: List[str],
                 ivfdata_fname: str) -> None:
    """ Add the contents of the indexes stored in shard_fnames into the index
    trained_index. The on-disk data is stored in ivfdata_fname """
    # merge the images into an on-disk index
    # first load the inverted lists
    ivfs = []
    for fname in shard_fnames:
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        #LOG.info("read " + fname)
        index = faiss.read_index(fname, faiss.IO_FLAG_MMAP)
        index_ivf = faiss.extract_index_ivf(index)
        ivfs.append(index_ivf.invlists)

        # avoid that the invlists get deallocated with the index
        index_ivf.own_invlists = False

    # construct the output index
    index = trained_index
    index_ivf = faiss.extract_index_ivf(index)

    assert index.ntotal == 0, "works only on empty index"

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(
        index_ivf.nlist, index_ivf.code_size,
        ivfdata_fname)

    # merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)

    #LOG.info("merge %d inverted lists " % ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

    # now replace the inverted lists in the output index
    index.ntotal = index_ivf.ntotal = ntotal
    index_ivf.replace_invlists(invlists, True)
    invlists.this.disown()


class FaissShardedOnDiskIdIndexer():
    '''
    Wraps a nn.module and calls forward_representation in forward (needed for multi-gpu use)
    '''

    def __init__(self,config,run_folder):
        super(FaissShardedOnDiskIdIndexer, self).__init__()
        self.token_dim=config["token_dim"]
        self.faiss_index_options= config["faiss_index_options"]
        self.run_folder = run_folder
        self.use_gpu = config["faiss_use_gpu"]

    def index_all(self, ids, data_chunks):
        shards = []
        #self.faiss_index = faiss.IndexShards(False,False)
        #
        # idea: we have one hub index, that is trained on the first shard, and then used to bucket the rest
        # all saved on disk
        #index = faiss.IndexIDMap(faiss.index_factory(self.token_dim,self.faiss_index_options,faiss.METRIC_INNER_PRODUCT))
        print("training centroids")
        res = faiss.StandardGpuResources()  
        index = faiss.index_factory(self.token_dim,self.faiss_index_options,faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.train(data_chunks[0].astype(numpy.float32))
        main_path=os.path.join(self.run_folder,"faiss_main.index")

        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index,main_path)
        del index
        gc.collect()
        print("go for indexing")
        
        for i, chunk in enumerate(data_chunks):

            #index = faiss.IndexIDMap(faiss.index_factory(self.token_dim,self.faiss_index_options,faiss.METRIC_INNER_PRODUCT))
            
            # vectors must be float32, ids must be int64
            #index.add_with_ids(chunk.astype(numpy.float32),numpy.array(ids[i],dtype="int64"))

            #shard_file=os.path.join(self.run_folder,"faiss_shard_"+str(i)+".index")
            #faiss.write_index(index, shard_file)

            #self.faiss_index.add_shard(faiss.read_index(shard_file,faiss.IO_FLAG_MMAP))

            index = faiss.read_index(main_path)

            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(res, 0, index)

            index.add_with_ids(chunk.astype(numpy.float32),numpy.array(ids[i],dtype="int64"))
            shard_file=os.path.join(self.run_folder,"faiss_shard_"+str(i)+".index")
            shards.append(shard_file)
            
            if self.use_gpu:
                index = faiss.index_gpu_to_cpu(index)

            faiss.write_index(index, shard_file)
            del index
            gc.collect()
            print("added",i)
        
        print("merging")

        merger = faiss.read_index(main_path)
        merge_ondisk(merger, shards, os.path.join(self.run_folder,"faiss_merged.index"))
        print("merged")

        faiss.write_index(merger, os.path.join(self.run_folder,"faiss_populated.index"))
        del merger
        gc.collect()

        #self.faiss_index = faiss.IndexShards(shards, successive_ids = False)
        self.faiss_index = faiss.read_index(os.path.join(self.run_folder,"faiss_populated.index"),faiss.IO_FLAG_MMAP)
        self.faiss_index.nprobe = 64
        print("index re-loaded")

    def load_prebuilt(self):
        #self.faiss_index = faiss.IndexShards(True,False)
        #for f in glob.glob(os.path.join(self.run_folder,"faiss_shard_*")):
        #    idx = faiss.read_index(f,faiss.IO_FLAG_MMAP)
        #    #print(idx.nprobe)
        #    idx.nprobe = 16
        #    self.faiss_index.add_shard(idx)
        #print("loaded shards:")

        self.faiss_index = faiss.read_index(os.path.join(self.run_folder,"faiss_populated.index"))
        self.faiss_index.nprobe = 16

    def search_single(self, query_vec,top_n):
        # even a single search must be 1xn dims
        if len(query_vec.shape) == 1:
            query_vec = query_vec[numpy.newaxis,:]
            
        res_scores, indices = self.faiss_index.search(query_vec.astype(numpy.float32),top_n)

        return res_scores, indices


def crappyhist(a, bins=20, width=30,range_=(0,1)):
    h, b = numpy.histogram(a, bins,range_)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i], 
            '#'*int(width*h[i]/numpy.amax(h)), 
            h[i],#/len(a), 
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

class FaissDynamicIndexer():
    '''
    Wraps an IVF (inverted list with centroid mapping) faiss index, and provides methods for continuous updates of centroids 
    '''

    def __init__(self,config):
        super(FaissDynamicIndexer, self).__init__()
        self.faiss_index = None
        #faiss.IndexIVFScalarQuantizer(faiss.IndexHNSWSQ(config["token_dim"], faiss.ScalarQuantizer.QT_fp16, 12, faiss.METRIC_INNER_PRODUCT),config["token_dim"],config["faiss_ivf_lists"],faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
        #self.faiss_index = faiss.IndexIVFFlat(config["token_dim"],config["faiss_ivf_lists"],faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
        
        #self.faiss_index = faiss.IndexIDMap(faiss.index_factory(config["token_dim"],config["faiss_index_options"],faiss.METRIC_INNER_PRODUCT))
        self.list_n_probe = 1
        self.token_dim = config["token_dim"]
        self.faiss_ivf_list_count = config["faiss_ivf_list_count"]

    def prepare(self, data_chunks:List[numpy.ndarray], subsample=-1):
        '''
        subsample = -1, train on all vecs, if 0 - 1 train on fraction rand subset
        '''
        total_vecs = sum(arr.shape[0] for arr in data_chunks)

        inv_list_count = int(self.faiss_ivf_list_count) # * numpy.sqrt(total_vecs))

        self.faiss_index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(self.token_dim),self.token_dim,inv_list_count,faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
        self.faiss_index.nprobe = self.list_n_probe

        if subsample > -1:
            max_train_sample = int(total_vecs * subsample)
            max_train_sample_per_chunk = max_train_sample // (len(data_chunks)-1)

            train_vecs = numpy.zeros((max_train_sample, data_chunks[0].shape[1]), dtype=numpy.float32)
            train_vecs[:,:] = 0

            rs = numpy.random.RandomState(123)
            for i, chunk in enumerate(data_chunks[:-1]):
                idx = rs.choice(chunk.shape[0], size=max_train_sample_per_chunk, replace=False)
                train_vecs[i*max_train_sample_per_chunk:(i+1)*max_train_sample_per_chunk] = chunk[idx]
        else:
            train_vecs = numpy.concatenate(data_chunks)
            max_train_sample = len(train_vecs)

        print("index with",total_vecs,"entries,",inv_list_count,"lists (avg-goal/list: ",total_vecs//inv_list_count,"); trained on",max_train_sample)

        self.faiss_index.train(train_vecs)


    def index_all(self, ids, data_chunks):
        for i, chunk in enumerate(data_chunks):
            #print(i,chunk.shape,ids[i].shape)
            # vectors must be float32, ids must be int64
            self.faiss_index.add_with_ids(chunk.astype(numpy.float32)[:len(ids[i])],numpy.array(ids[i],dtype="int64"))

        self.faiss_index.invlists.print_stats()
        print("imbalance factor",self.faiss_index.invlists.imbalance_factor())
        sizes = [self.faiss_index.invlists.list_size(i) for i in range(0,self.faiss_index.invlists.nlist)]
        print("median:",numpy.median(sizes))
        print("initial distibution")
        crappyhist(sizes,range_=(0,max(sizes)))

    def update(self, ids, data):

        ids = numpy.array(ids,dtype="int64")

        self.faiss_index.remove_ids(ids)
        
        # vectors must be float32, ids must be int64
        self.faiss_index.add_with_ids(data.astype(numpy.float32)[:len(ids)],ids)

        self.faiss_index.invlists.print_stats()
        print("imbalance factor",self.faiss_index.invlists.imbalance_factor())
        #sizes = [self.faiss_index.invlists.list_size(i) for i in range(0,self.faiss_index.invlists.nlist)]
        #print("updated distibution")
        #crappyhist(sizes,range_=(0,max(sizes)))

    def get_entries_from_centroids(self,centroid_ids):
        entry_ids = []
        for list_no in centroid_ids:
            list_sz = self.faiss_index.invlists.list_size(int(list_no))  # The length of list_no-th posting list

            # Fetch
            id_poslist = numpy.array(faiss.rev_swig_ptr(self.faiss_index.invlists.get_ids(int(list_no)), list_sz))

            entry_ids.extend(list(id_poslist))
        
        return entry_ids

    def get_all_cluster_assignments(self):
        clusters=[]
        for c in range(self.faiss_index.nlist):
            clusters.append(self.get_entries_from_centroids([c]))
        return clusters

    def search_single(self, query_vec,top_n):
        # even a single search must be 1xn dims
        if len(query_vec.shape) == 1:
            query_vec = query_vec[numpy.newaxis,:]
        
        query_vec = query_vec.astype(numpy.float32)

        #
        # get the nearest centroids (inverted list ids) first, so we know which centroid ids are touched
        #
        centroid_dist, centroid_ids = self.faiss_index.quantizer.search(query_vec,self.list_n_probe)

        #
        # search only on the centroid_ids we already have (saves to do that part twice)
        #
        numpy_res_ind = numpy.zeros((len(query_vec), top_n), dtype=numpy.int64)
        numpy_res_dist = numpy.zeros((len(query_vec), top_n), dtype=numpy.float32)
        self.faiss_index.search_preassigned(len(query_vec),faiss.swig_ptr(query_vec),
                                            top_n,
                                            faiss.swig_ptr(centroid_ids),faiss.swig_ptr(centroid_dist),
                                            faiss.swig_ptr(numpy_res_dist),faiss.swig_ptr(numpy_res_ind),
                                            False,None)

        #invalidated_ids = self.get_entries_from_centroids(numpy_res_ind[0])
        # old direct search
        #res_scores, indices =  self.faiss_index.search(query_vec.astype(numpy.float32),top_n)

        return numpy_res_dist, numpy_res_ind, centroid_ids


    # coarse assignment
    #coarse_dis, assign = index.quantizer.search(xq, index.nprobe)
    #nlist = index.nlist
    #assign_buckets = assign // bs
    #nq = len(xq)
#
    #rh = faiss.ResultHeap(nq, k)
    #index.parallel_mode |= index.PARALLEL_MODE_NO_HEAP_INIT
#
    #for l0 in range(0, nlist, bs):
    #    bucket_no = l0 // bs
    #    skip_rows, skip_cols = np.where(assign_buckets != bucket_no)
    #    sub_assign = assign.copy()
    #    sub_assign[skip_rows, skip_cols] = -1
#
    #    index.search_preassigned(
    #        nq, faiss.swig_ptr(xq), k,
    #        faiss.swig_ptr(sub_assign), faiss.swig_ptr(coarse_dis),
    #        faiss.swig_ptr(rh.D), faiss.swig_ptr(rh.I),
    #        False, None
    #    )
#
    #rh.finalize()
#
    #return rh.D, rh.I