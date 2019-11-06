import torch.multiprocessing as mp

import torch
import numpy 
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from dataloaders.fasttext_token_indexer import *
from dataloaders.ir_triple_loader import *
from dataloaders.ir_labeled_tuple_loader import IrLabeledTupleDatasetReader
from matchmaker.dataloaders.ir_single_sequence_loader import *

from dataloaders.bert_triple_loader import *
from dataloaders.bert_labeled_tuple_loader import *
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer

from typing import Dict, Tuple, List

#
# Multiprocess input pipeline
# -------------------------------
#
# single epoch batch generators with multiple subprocesses, each subprocess works on its own file until the file is parsed completely
#
# - the processes have as little communication as possible (because it is prohibitly expensive in python)
# - the finished batches go into shared memory and then the queue to be picked up by the train/validaton loops
#

mp.get_logger().setLevel(logging.WARNING)  # ignore useless process start console logs
mp.set_sharing_strategy("file_system") # VERY MUCH needed for linux !! makes everything MUCH faster -> from 10 to 30+ batches/s

fasttext_vocab_cached_mapping = None
fasttext_vocab_cached_data = None

#
# we need to wrap the individual process queues, because they might be filled in different order 
# now we make sure to always get the same training samples in the same order for all runs
#
class DeterministicQueue():
    def __init__(self,
                 distributed_queues):
        self.distributed_queues = distributed_queues
        self.num_queues = len(distributed_queues)
        self.current_idx = 0

    def get(self):
        element = self.distributed_queues[self.current_idx].get()
        if element == None:
            self.distributed_queues.pop(self.current_idx) # this queue is done 
            self.num_queues = len(self.distributed_queues)
        else:
            self.current_idx += 1
        
        if self.current_idx >= self.num_queues:
            self.current_idx = 0

        if element == None: # not guaranteed that queues have the same number of elements
            if self.num_queues == 0: # was this truly the last queue?
                return None
            element = self.get()

        return element

    def qsize(self):
        size = 0
        for q in self.distributed_queues:
            size+=q.qsize()
        return size

#
# process & queue starter, returns a queue which gets the batches put into ready to go into the model.forward pass
#
def get_multiprocess_batch_queue(name_prefix: str, target_function, files, conf, _logger, queue_size=100) -> Tuple[mp.Queue, List[mp.Process], mp.Event]:
    ctx = mp.get_context('spawn') # also set so that windows & linux behave the same 
    _processes = []
    _finish_notification = ctx.Event()

    if len(files) == 0:
        _logger.error("No files for multiprocess loading specified, for: " + name_prefix)
        exit(1)
    else:
        _logger.info("Starting "+str(len(files))+" data loader processes, for:" + name_prefix)

    if conf["token_embedder_type"] == "fasttext":
        global fasttext_vocab_cached_mapping
        global fasttext_vocab_cached_data
        if fasttext_vocab_cached_data is None:
            fasttext_vocab_cached_mapping, fasttext_vocab_cached_data = FastTextVocab.load_ids(conf["fasttext_vocab_mapping"],conf["fasttext_max_subwords"])
            fasttext_vocab_cached_data.share_memory_()

    _queue_list = []
    #_queue = ctx.Queue(queue_size)
    for proc_number, file in enumerate(files):
        _queue = ctx.Queue(queue_size)
        process = ctx.Process(name=name_prefix + "-" + str(proc_number),
                             target=target_function,
                             args=(proc_number, conf, _queue, _finish_notification, file,fasttext_vocab_cached_mapping,fasttext_vocab_cached_data))
        process.start()
        _processes.append(process)
        _queue_list.append(_queue)
    return DeterministicQueue(_queue_list), _processes, _finish_notification
    #return _queue, _processes, _finish_notification


#
# training instance generator
#   - filling the _queue with ready to run training batches
#   - everything is thread local
#
def multiprocess_training_loader(process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):

    torch.manual_seed(_config["random_seed"])
    numpy.random.seed(_config["random_seed"])
    random.seed(_config["random_seed"])

    if _config["token_embedder_type"] == "bert_cls":
        _tokenizer = BlingFireTokenizer()
        _ind = PretrainedBertIndexer(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True)
        _token_indexers = {"tokens": _ind}

        _triple_loader = BertTripleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
                                               max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
                                               min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"])
    
        _iterator = BucketIterator(batch_size=int(_config["batch_size_train"]),
                                   sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])
    
        _iterator.index_with(Vocabulary())#.from_files(_config["vocab_directory"]))

    else:
        _tokenizer = BlingFireTokenizer()

        if _config["token_embedder_type"] == "embedding":
            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
            _vocab = Vocabulary.from_files(_config["vocab_directory"])

        elif _config["token_embedder_type"] == "fasttext":
            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])

        elif _config["token_embedder_type"] == "elmo":
            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
            _vocab = None

        _triple_loader = IrTripleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
                                               max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
                                               min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"])

        _iterator = BucketIterator(batch_size=int(_config["batch_size_train"]),
                                   sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

        _iterator.index_with(_vocab)

    for training_batch in _iterator(_triple_loader.read(_local_file), num_epochs=1):

        _queue.put(training_batch)  # this moves the tensors in to shared memory

    _queue.put(None) # signal end of queue

    _queue.close()  # indicate this local thread is done
    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore

#
# validation instance generator
#   - filling the _queue with ready to run validation batches
#   - everything is defined thread local
#
def multiprocess_validation_loader(process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):

    torch.manual_seed(_config["random_seed"])
    numpy.random.seed(_config["random_seed"])
    random.seed(_config["random_seed"])

    if _config["token_embedder_type"] == "bert_cls":
        _tokenizer = BlingFireTokenizer()
        _ind = PretrainedBertIndexer(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True)
        _token_indexers = {"tokens": _ind}

        _tuple_loader = BertLabeledTupleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
                                                      max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
                                                      min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"])
    
        _iterator = BucketIterator(batch_size=int(_config["batch_size_eval"]),
                                   sorting_keys=[("doc_tokens", "num_tokens")])
    
        _iterator.index_with(Vocabulary.from_files(_config["vocab_directory"]))

    else:
        _tokenizer = BlingFireTokenizer()

        if _config["token_embedder_type"] == "embedding":
            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
            _vocab = Vocabulary.from_files(_config["vocab_directory"])

        elif _config["token_embedder_type"] == "fasttext":
            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])

        elif _config["token_embedder_type"] == "elmo":
            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
            _vocab = None

        _tuple_loader = IrLabeledTupleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
                                                    max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
                                                    min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"])

        _iterator = BucketIterator(batch_size=int(_config["batch_size_eval"]),
                                   sorting_keys=[("doc_tokens", "num_tokens"), ("query_tokens", "num_tokens")])

        _iterator.index_with(_vocab)

    for training_batch in _iterator(_tuple_loader.read(_local_file), num_epochs=1):

        _queue.put(training_batch)  # this moves the tensors in to shared memory

    _queue.put(None) # signal end of queue

    _queue.close()  # indicate this local thread is done
    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore

#
# single sequence loader from multiple files 
#
def multiprocess_single_sequence_loader(process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):

    torch.manual_seed(_config["random_seed"])
    numpy.random.seed(_config["random_seed"])
    random.seed(_config["random_seed"])

    if _config["token_embedder_type"] == "bert_cls":
        _tokenizer = BlingFireTokenizer()
        _ind = PretrainedBertIndexer(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True)
        _token_indexers = {"tokens": _ind}

        _tuple_loader = IrSingleSequenceDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
                                                max_seq_length= _config["max_doc_length"], min_seq_length=_config["min_doc_length"],)

        _iterator = BucketIterator(batch_size=int(_config["batch_size_eval"]),
                                   sorting_keys=[("seq_tokens", "num_tokens")])

        _iterator.index_with(Vocabulary.from_files(_config["vocab_directory"]))

    else:
        _tokenizer = BlingFireTokenizer()

        if _config["token_embedder_type"] == "embedding":
            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
            _vocab = Vocabulary.from_files(_config["vocab_directory"])

        elif _config["token_embedder_type"] == "fasttext":
            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])

        elif _config["token_embedder_type"] == "elmo":
            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
            _vocab = None

        _tuple_loader = IrSingleSequenceDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
                                                    max_seq_length= _config["max_doc_length"], min_seq_length=_config["min_doc_length"],)

        _iterator = BucketIterator(batch_size=int(_config["batch_size_eval"]),
                                   sorting_keys=[("seq_tokens", "num_tokens")])

        _iterator.index_with(_vocab)

    for training_batch in _iterator(_tuple_loader.read(_local_file), num_epochs=1):

        _queue.put(training_batch)  # this moves the tensors in to shared memory

    _queue.put(None) # signal end of queue

    _queue.close()  # indicate this local thread is done
    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore