from rich.console import Console
import random
from matchmaker.dataloaders.transformer_tokenizer import *
from allennlp.data.batch import Batch
from matchmaker.utils.core_metrics import *
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField
import numpy as np
from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.data.data_loaders.multiprocess_data_loader import WorkerError
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterator, List
import logging
import torch
import torch.multiprocessing as mp
#mp.set_sharing_strategy("file_system")  # VERY MUCH needed for linux !! makes everything MUCH faster


class TASBalancedDatasetLoader():
    """
    dynamically samples queries from given cluster information for a batch
    """

    def __init__(
        self,

        query_file: str,
        collection_file: str,
        pairs_with_teacher_scores: str,
        query_cluster_file: str,
        batch_size: int,
        clusters_per_batch: str,

        tokenizer: Tokenizer = None,

        max_doc_length: int = -1,
        max_query_length: int = -1,

        pair_balancing_strategy="bins",  # or "random" or "hard-margin"

        random_seed=42,
    ):

        self.query_file = query_file
        self.collection_file = collection_file
        self.pairs_with_teacher_scores = pairs_with_teacher_scores
        self.query_cluster_file = query_cluster_file
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch

        self._tokenizer = tokenizer

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        if type(tokenizer) != FastTransformerTokenizer:
            raise Exception("only huggingface tokenizer supported")

        self.read_with_scores = True

        self.pair_balancing_strategy = pair_balancing_strategy
        self.uniform_percentile_sampling = pair_balancing_strategy == "bins"
        self.uniform_percentile_sampling_bins = 10

        #self.hard_margin_sampling = pair_balancing_strategy == "hard-margin"
        #self.hard_margin_sampling_cutoff = 6

        #margins = []
        self.seed = random_seed

    def __iter__(self) -> Iterator[TensorDict]:
        
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")

        queue: mp.JoinableQueue = ctx.JoinableQueue(2000)
        worker = ctx.Process(
            target=self.data_loader_subprocess, args=(queue,), daemon=True
        )
        worker.start()

        try:
            for batch, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise WorkerError(e, tb)

                yield batch
                queue.task_done()
        finally:
            if hasattr(queue, "close"):  # for compat with different Python versions.
                queue.close()  # type: ignore[attr-defined]
            if worker.is_alive():
                worker.terminate()

    def load_data(self):

        console = Console()

        console.log("[TASBalanced] Loading collection from:",self.collection_file)
        self.collection = {}
        self.collection_ids = []
        with open(self.collection_file, "r", encoding="utf8") as cf:
            for line in cf:
                ls = line.split("\t")  # id<\t>text ....
                self.collection[ls[0]] = ls[1].rstrip()[:100_000]
                self.collection_ids.append(ls[0])

        console.log("[TASBalanced] Loading queries from:",self.query_file)
        self.queries = {}
        with open(self.query_file, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split("\t")  # id<\t>text ....
                self.queries[ls[0]] = ls[1].rstrip()

        console.log("[TASBalanced] Loading pairs from:",self.pairs_with_teacher_scores)
        self.pairs_with_teacher_scores_by_qid = defaultdict(list)
        with open(self.pairs_with_teacher_scores, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split()  # pos_score<t>neg_score<t>pos_id<t>neg_id
                # margins.append(float(ls[0])-float(ls[1]))
                # if self.hard_margin_sampling and float(ls[0])-float(ls[1]) > self.hard_margin_sampling_cutoff:
                #    continue
                self.pairs_with_teacher_scores_by_qid[ls[2]].append((ls[3], ls[4].strip(), float(ls[0]), float(ls[1])))

        # print("margin-mean",np.mean(margins))
        # print("margin-median",np.median(margins))

        if self.uniform_percentile_sampling:
            console.log("[TASBalanced] Creating balanced bins")
            pairs_with_teacher_scores_by_qid_binned = defaultdict(list)
            avg_bin_lengths = [[] for _ in range(self.uniform_percentile_sampling_bins)]
            for q_id, pair_list in self.pairs_with_teacher_scores_by_qid.items():
                if len(pair_list) >= 2:
                    margins = np.array([l[2] - l[3] for l in pair_list])
                    indices = np.digitize(margins, np.arange(np.min(margins), np.max(margins), (np.max(margins)-np.min(margins))/self.uniform_percentile_sampling_bins))
                    bins = [[] for _ in range(self.uniform_percentile_sampling_bins)]
                    for i, p in enumerate(pair_list):
                        bins[indices[i]-1].append(p)
                    for i, b in enumerate(bins):
                        avg_bin_lengths[i].append(len(b))
                    pairs_with_teacher_scores_by_qid_binned[q_id] = bins
            #for i, b in enumerate(avg_bin_lengths):
            #    print("bin", i, "avg:", np.mean(b),"num == 0",np.sum(np.array(b) == 0))
            self.pairs_with_teacher_scores_by_qid = pairs_with_teacher_scores_by_qid_binned

        console.log("[TASBalanced] Loading cluster assignments from:",self.query_cluster_file)
        self.query_clusters = []
        all_cluster_ids = []
        with open(self.query_cluster_file, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split()  # id<\t>id ....
                self.query_clusters.append(ls)
                all_cluster_ids.extend(ls)

        self.query_ids = set(self.pairs_with_teacher_scores_by_qid.keys()).intersection(set(all_cluster_ids))

        # clean clusters, to only have matching ids with pair file
        for i, c in enumerate(self.query_clusters):
            self.query_clusters[i] = list(set(c).intersection(self.query_ids))
        self.query_clusters = [c for c in self.query_clusters if len(c) > 0]

        console.log("[TASBalanced] Done loading! Using ", len(self.query_ids), " queries from ", len(self.query_clusters),"clusters for seed:",self.seed," with pair_balancing_strategy: ",self.pair_balancing_strategy)

    def data_loader_subprocess(self, queue):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        try:
            self.load_data()

            query_target_count = int((self.batch_size / self.clusters_per_batch))

            while True:

                main_instances = []

                while len(main_instances) < self.batch_size:

                    # get rnd cluster
                    c_idx = random.randint(0, len(self.query_clusters)-1)

                    # take a query sample out of that cluster
                    if query_target_count < len(self.query_clusters[c_idx]):
                        q_ids = random.sample(self.query_clusters[c_idx], query_target_count)
                    else:
                        q_ids = self.query_clusters[c_idx]

                    for q_id in q_ids:
                        query_text = self.get_tokenized_query(self.queries[q_id])

                        if self.uniform_percentile_sampling:
                            pair = None
                            while pair == None:
                                bin_idx = random.randint(0, len(self.pairs_with_teacher_scores_by_qid[q_id])-1)
                                if len(self.pairs_with_teacher_scores_by_qid[q_id][bin_idx]) > 0:
                                    pair = random.choice(self.pairs_with_teacher_scores_by_qid[q_id][bin_idx])

                        else:
                            pair = random.choice(self.pairs_with_teacher_scores_by_qid[q_id])

                        pos_text = self.get_tokenized_document(self.collection[pair[0]])
                        neg_text = self.get_tokenized_document(self.collection[pair[1]])

                        ret_instance = {
                            "query_tokens":     query_text,
                            "doc_pos_tokens":   pos_text,
                            "doc_neg_tokens":   neg_text,
                            # "cluster_idx":      MetadataField(np.array(c_idx))
                        }

                        if self.read_with_scores:
                            ret_instance["pos_score"] = ArrayField(np.array(pair[2]))
                            ret_instance["neg_score"] = ArrayField(np.array(pair[3]))

                        main_instances.append(Instance(ret_instance))

                        if len(main_instances) == self.batch_size:
                            break

                main_batch = Batch(main_instances)
                main_batch = main_batch.as_tensor_dict(main_batch.get_padding_lengths())

                queue.put((main_batch,None))

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))
        
        queue.put((None, None))
        # Wait until this process can safely exit.
        queue.join()

    def get_tokenized_query(self, text):
        query_tokenized = self._tokenizer.tokenize(text, max_length=self.max_query_length)
        return PatchedTransformerTextField(**query_tokenized)

    def get_tokenized_document(self, text):
        doc_tokenized = self._tokenizer.tokenize(text, max_length=self.max_doc_length)
        return PatchedTransformerTextField(**doc_tokenized)