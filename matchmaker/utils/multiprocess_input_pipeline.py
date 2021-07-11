#import torch.multiprocessing as mp
#
#import torch
#import numpy 
#import random
#
#from allennlp.data.samplers import BucketBatchSampler
#from allennlp.data.vocabulary import Vocabulary
#from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
#from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
#
#from matchmaker.dataloaders.fasttext_token_indexer import *
#from matchmaker.dataloaders.ir_triple_loader import *
#from matchmaker.dataloaders.ir_labeled_tuple_loader import IrLabeledTupleDatasetReader
#from matchmaker.dataloaders.ir_single_sequence_loader import *
#from matchmaker.dataloaders.mlm_masked_sequence_loader import *
#from matchmaker.dataloaders.dynamic_triple_loader import *
#
#from matchmaker.dataloaders.bert_triple_loader import *
#from matchmaker.dataloaders.bert_labeled_tuple_loader import *
#from allennlp.data.token_indexers import PretrainedTransformerIndexer
#from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer
#
#from matchmaker.modules.bert_embedding_token_embedder import PretrainedBertIndexerNoSpecialTokens
#from allennlp.data.tokenizers import PretrainedTransformerTokenizer
#
#from typing import Dict, Tuple, List
##from tokenizers import ByteLevelBPETokenizer,CharBPETokenizer
#from matchmaker.dataloaders.transformer_tokenizer import CustomTransformerTokenizer,CustomTransformerIndexer
#
#from allennlp.data.data_loaders import MultiProcessDataLoader
#
##
## Multiprocess input pipeline
## -------------------------------
##
## single epoch batch generators with multiple subprocesses, each subprocess works on its own file until the file is parsed completely
##
## - the processes have as little communication as possible (because it is prohibitly expensive in python)
## - the finished batches go into shared memory and then the queue to be picked up by the train/validaton loops
##
#
#mp.get_logger().setLevel(logging.WARNING)  # ignore useless process start console logs
#mp.set_sharing_strategy("file_system") # VERY MUCH needed for linux !! makes everything MUCH faster -> from 10 to 30+ batches/s
#
#fasttext_vocab_cached_mapping = None
#fasttext_vocab_cached_data = None
#
##
## we need to wrap the individual process queues, because they might be filled in different order 
## now we make sure to always get the same training samples in the same order for all runs
##
#class DeterministicQueue():
#    def __init__(self,
#                 distributed_queues):
#        self.distributed_queues = distributed_queues
#        self.num_queues = len(distributed_queues)
#        self.current_idx = 0
#
#    def get(self):
#        element = self.distributed_queues[self.current_idx].get()
#        if element == None:
#            self.distributed_queues.pop(self.current_idx) # this queue is done 
#            self.num_queues = len(self.distributed_queues)
#        else:
#            self.current_idx += 1
#        
#        if self.current_idx >= self.num_queues:
#            self.current_idx = 0
#
#        if element == None: # not guaranteed that queues have the same number of elements
#            if self.num_queues == 0: # was this truly the last queue?
#                return None
#            element = self.get()
#
#        return element
#
#    def qsize(self):
#        size = 0
#        for q in self.distributed_queues:
#            size+=q.qsize()
#        return size
#
##
## process & queue starter, returns a queue which gets the batches put into ready to go into the model.forward pass
##
#def get_multiprocess_batch_queue(epoch: int, target_function, files, conf, _logger, queue_size=100,sequence_type="doc") -> Tuple[mp.Queue, List[mp.Process], mp.Event]:
#    ctx = mp.get_context('spawn') # also set so that windows & linux behave the same 
#    _processes = []
#    _finish_notification = ctx.Event()
#
#    if len(files) == 0:
#        _logger.error("No files for multiprocess loading specified, for: " + str(epoch))
#        exit(1)
#    else:
#        _logger.info("Starting "+str(len(files))+" data loader processes, for:" + str(epoch))
#
#    if conf["token_embedder_type"] == "fasttext":
#        global fasttext_vocab_cached_mapping
#        global fasttext_vocab_cached_data
#        if fasttext_vocab_cached_data is None:
#            fasttext_vocab_cached_mapping, fasttext_vocab_cached_data = FastTextVocab.load_ids(conf["fasttext_vocab_mapping"],conf["fasttext_max_subwords"])
#            fasttext_vocab_cached_data.share_memory_()
#
#    _queue_list = []
#    #_queue = ctx.Queue(queue_size)
#    for proc_number, file in enumerate(files):
#        _queue = ctx.Queue(queue_size)
#        process = ctx.Process(name=str(epoch) + "-" + str(proc_number),
#                             target=target_function,
#                             args=(epoch,proc_number, conf, _queue, _finish_notification, file, sequence_type, fasttext_vocab_cached_mapping, fasttext_vocab_cached_data))
#        process.start()
#        _processes.append(process)
#        _queue_list.append(_queue)
#    return DeterministicQueue(_queue_list), _processes, _finish_notification
#    #return _queue, _processes, _finish_notification
#
##
## process & queue starter, returns a queue which gets the batches put into ready to go into the model.forward pass
##
#def get_multiprocess_batch_queue_dynamic(epoch: int, target_function, conf, _logger, queue_size=100,sequence_type="doc") -> Tuple[mp.Queue, List[mp.Process], mp.Event]:
#    ctx = mp.get_context('spawn') # also set so that windows & linux behave the same 
#    _processes = []
#    _finish_notification = ctx.Event()
#    _logger.info("Starting "+str(conf["dynamic_workers"])+" data loader processes, for:" + str(epoch))
#    _queue_list = []
#    #_queue = ctx.Queue(queue_size)
#    for proc_number in range(conf["dynamic_workers"]):
#        _queue = ctx.Queue(queue_size)
#        process = ctx.Process(name=str(epoch) + "-" + str(proc_number),
#                             target=target_function,
#                             args=(epoch,proc_number, conf, _queue, _finish_notification, None,sequence_type,None,None))
#        process.start()
#        _processes.append(process)
#        _queue_list.append(_queue)
#    return DeterministicQueue(_queue_list), _processes, _finish_notification
#
##
## training instance generator
##   - filling the _queue with ready to run training batches
##   - everything is thread local
##
#def multiprocess_training_loader(epoch: int, process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,sequence_type,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):
#
#    if _config["dynamic_sampler"] == True:
#        torch.manual_seed(_config["random_seed"] + 1000 * process_number)
#        numpy.random.seed(_config["random_seed"] + 1000 * process_number)
#        random.seed(_config["random_seed"] + 1000 * process_number)
#    else:
#        torch.manual_seed(_config["random_seed"])
#        numpy.random.seed(_config["random_seed"])
#        random.seed(_config["random_seed"])
#   
#    if _config["token_embedder_type"] == "bert_cat":
#        _tokenizer = PretrainedTransformerTokenizer(_config["bert_pretrained_model"])
#        _ind = PretrainedTransformerIndexer(model_name=_config["bert_pretrained_model"])
#        _token_indexers = {"tokens": _ind}
#
#        _triple_loader = BertTripleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                               max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
#                                               min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"],
#                                               data_augment=_config["train_data_augment"],train_pairwise_distillation=_config["train_pairwise_distillation"],train_qa_spans=_config["train_qa_spans"])
#    
#        data_reader = _triple_loader.read(_local_file)
#        data_reader.index_with(Vocabulary())
#        loader = SimpleDataLoader(data_reader, batch_size=int(_config["batch_size_train"]))
#    else:
#        _tokenizer = BlingFireTokenizer()
#
#        if _config["token_embedder_type"] == "embedding":
#            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
#            _vocab = Vocabulary.from_files(_config["vocab_directory"])
#
#        elif _config["token_embedder_type"] == "fasttext":
#            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
#            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])
#
#        elif _config["token_embedder_type"] == "elmo":
#            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
#            _vocab = None
#
#        elif _config["token_embedder_type"] == "bert_embedding" or _config["token_embedder_type"] == "bert_vectors":
#            _tokenizer = PretrainedTransformerTokenizer(_config["bert_pretrained_model"], do_lowercase=True,start_tokens =[],end_tokens=[])               
#            _ind = PretrainedBertIndexerNoSpecialTokens(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True,max_pieces=_config["max_doc_length"])
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        elif _config["token_embedder_type"] == "bert_dot":
#            model = _config["bert_pretrained_model"]
#            if "facebook/dpr" in _config["bert_pretrained_model"]: # ugh .. they forgot to upload the tokenizer
#                model = "bert-base-uncased"                        # should be identical (judging from paper + huggingface doc)
#            _tokenizer = PretrainedTransformerTokenizer(model)               
#            _ind = PretrainedTransformerIndexer(model_name=model)
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        elif _config["token_embedder_type"] == "huggingface_bpe":
#
#            files = _config["bpe_vocab_files"].split(";")
#
#            _ind = CustomTransformerIndexer(CharBPETokenizer(files[0],files[1], lowercase=True))
#            _tokenizer = _ind._allennlp_tokenizer
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        if _config["dynamic_sampler"] == True:
#            if _config["dynamic_sampler_type"] == "list":
#                loader = IrDynamicTripleDatasetLoader(query_file=_config["dynamic_query_file"],collection_file=_config["dynamic_collection_file"],
#                                                  qrels_file=_config["dynamic_qrels_file"],candidate_file=_config["dynamic_candidate_file"],
#                                                  batch_size=int(_config["batch_size_train"]),queries_per_batch=_config["dynamic_queries_per_batch"], tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                  max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
#                                                  min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"],
#                                                  data_augment=_config["train_data_augment"],vocab=_vocab)
#            elif _config["dynamic_sampler_type"] == "cluster":
#                loader = TASBalancedDatasetLoader(query_file=_config["dynamic_query_file"],collection_file=_config["dynamic_collection_file"],
#                                                  pairs_with_teacher_scores=_config["dynamic_pairs_with_teacher_scores"],query_cluster_file=_config["dynamic_query_cluster_file"],
#                                                  batch_size=int(_config["batch_size_train"]),clusters_per_batch=_config["dynamic_clusters_per_batch"], tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                  max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
#                                                  min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"],
#                                                  data_augment=_config["train_data_augment"],vocab=_vocab,sampling_strategy=_config["tas_balanced_pair_strategy"])
#        else:
#            _triple_loader = IrTripleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                   max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
#                                                   min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"],
#                                                   data_augment=_config["train_data_augment"],train_pairwise_distillation=_config["train_pairwise_distillation"],
#                                                   query_augment_mask_number=_config["query_augment_mask_number"],train_qa_spans=_config["train_qa_spans"])
#
#            #data_reader = _triple_loader.read(_local_file)
#            #data_reader.index_with(_vocab)
#            #loader = SimpleDataLoader(data_reader, batch_size=int(_config["batch_size_train"]))
#
#            loader = MultiProcessDataLoader(_triple_loader,data_path=_local_file, batch_size=int(_config["batch_size_train"]),max_instances_in_memory=10000,quiet=True)
#            loader.index_with(_vocab)
#
#    for training_batch in loader:
#
#        _queue.put(training_batch)  # this moves the tensors in to shared memory
#
#    _queue.put(None) # signal end of queue
#
#    _queue.close()  # indicate this local thread is done
#    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore
#
##
## validation instance generator
##   - filling the _queue with ready to run validation batches
##   - everything is defined thread local
##
#def multiprocess_validation_loader(epoch: int, process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,sequence_type,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):
#
#    torch.manual_seed(_config["random_seed"])
#    numpy.random.seed(_config["random_seed"])
#    random.seed(_config["random_seed"])
#
#    if _config["token_embedder_type"] == "bert_cat":
#        _tokenizer = PretrainedTransformerTokenizer(_config["bert_pretrained_model"])
#        _ind = PretrainedTransformerIndexer(model_name=_config["bert_pretrained_model"])
#        _token_indexers = {"tokens": _ind}
#
#        _tuple_loader = BertLabeledTupleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                      max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
#                                                      min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"],train_qa_spans=_config["train_qa_spans"])
#        _vocab = Vocabulary()
#
#    else:
#        _tokenizer = BlingFireTokenizer()
#
#        if _config["token_embedder_type"] == "embedding":
#            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
#            _vocab = Vocabulary.from_files(_config["vocab_directory"])
#
#        elif _config["token_embedder_type"] == "fasttext":
#            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
#            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])
#
#        elif _config["token_embedder_type"] == "elmo":
#            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
#            _vocab = None
#        elif _config["token_embedder_type"] == "bert_embedding" or _config["token_embedder_type"] == "bert_vectors":
#            _tokenizer = PretrainedTransformerTokenizer(_config["bert_pretrained_model"], do_lowercase=True,start_tokens =[],end_tokens=[])
#            _ind = PretrainedBertIndexerNoSpecialTokens(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True,max_pieces=_config["max_doc_length"])
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        elif _config["token_embedder_type"] == "bert_dot":
#            model = _config["bert_pretrained_model"]
#            if "facebook/dpr" in _config["bert_pretrained_model"]: # ugh .. they forgot to upload the tokenizer
#                model= "bert-base-uncased"                        # should be identical (judging from paper + huggingface doc)
#            _tokenizer = PretrainedTransformerTokenizer(model)               
#            _ind = PretrainedTransformerIndexer(model_name=model)
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        elif _config["token_embedder_type"] == "huggingface_bpe":
#
#            files = _config["bpe_vocab_files"].split(";")
#
#            _ind = CustomTransformerIndexer(CharBPETokenizer(files[0],files[1], lowercase=True))
#            _tokenizer = _ind._allennlp_tokenizer
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        _tuple_loader = IrLabeledTupleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                    max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"],
#                                                    min_doc_length=_config["min_doc_length"],min_query_length=_config["min_query_length"],
#                                                    query_augment_mask_number=_config["query_augment_mask_number"],train_qa_spans=_config["train_qa_spans"])
#
#    #data_reader = _tuple_loader.read(_local_file)
#    #data_reader.index_with(_vocab)
#    #loader = SimpleDataLoader(data_reader, batch_size=int(_config["batch_size_eval"]))
#
#    loader = MultiProcessDataLoader(_tuple_loader,data_path=_local_file, batch_size=int(_config["batch_size_eval"]),max_instances_in_memory=10000,quiet=True)
#    loader.index_with(_vocab)
#
#
#    for training_batch in loader:
#
#        _queue.put(training_batch)  # this moves the tensors in to shared memory
#
#    _queue.put(None) # signal end of queue
#
#    _queue.close()  # indicate this local thread is done
#    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore
#
##
## single sequence loader from multiple files 
##
#def multiprocess_single_sequence_loader(epoch: int, process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file, sequence_type,
#                                        _fasttext_vocab_cached_mapping, _fasttext_vocab_cached_data):
#
#    torch.manual_seed(_config["random_seed"])
#    numpy.random.seed(_config["random_seed"])
#    random.seed(_config["random_seed"])
#
#    if sequence_type == "query":
#        max_length = _config["max_query_length"]
#        min_length = _config["min_query_length"]
#
#    else: # doc
#        max_length = _config["max_doc_length"]
#        min_length = _config["min_doc_length"]
#
#    if _config["token_embedder_type"] == "bert_cat":
#        _tokenizer = BlingFireTokenizer()
#        _ind = PretrainedTransformerIndexer(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True)
#        _token_indexers = {"tokens": _ind}
#
#        _tuple_loader = IrSingleSequenceDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                      max_seq_length= max_length, min_seq_length=min_length,)
#
#    else:
#        _tokenizer = BlingFireTokenizer()
#
#        if _config["token_embedder_type"] == "embedding":
#            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
#            _vocab = Vocabulary.from_files(_config["vocab_directory"])
#
#        elif _config["token_embedder_type"] == "fasttext":
#            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
#            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])
#
#        elif _config["token_embedder_type"] == "elmo":
#            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
#            _vocab = None
#        elif _config["token_embedder_type"] == "bert_embedding" or _config["token_embedder_type"] == "bert_vectors":
#            _tokenizer = PretrainedTransformerTokenizer(_config["bert_pretrained_model"], do_lowercase=True,start_tokens =[],end_tokens=[])
#            _ind = PretrainedBertIndexerNoSpecialTokens(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True,max_pieces=max_length)
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#        elif _config["token_embedder_type"] == "bert_dot":
#            model = _config["bert_pretrained_model"]
#            if "facebook/dpr" in _config["bert_pretrained_model"]: # ugh .. they forgot to upload the tokenizer
#                model= "bert-base-uncased"                        # should be identical (judging from paper + huggingface doc)
#            _tokenizer = PretrainedTransformerTokenizer(model)               
#            _ind = PretrainedTransformerIndexer(model_name=model)            
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#        elif _config["token_embedder_type"] == "huggingface_bpe":
#
#            files = _config["bpe_vocab_files"].split(";")
#
#            _ind = CustomTransformerIndexer(ByteLevelBPETokenizer(files[0],files[1]))
#            _tokenizer = _ind._allennlp_tokenizer
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        _tuple_loader = IrSingleSequenceDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                      max_seq_length= max_length, min_seq_length=min_length,sequence_type=sequence_type)
#
#    #data_reader = _tuple_loader.read(_local_file)
#    #data_reader.index_with(_vocab)
#    loader = MultiProcessDataLoader(_tuple_loader,data_path=_local_file, batch_size=int(_config["batch_size_eval"]),max_instances_in_memory=10000,quiet=True)
#    loader.index_with(_vocab)
#    for training_batch in loader:
#        _queue.put(training_batch)  # this moves the tensors in to shared memory
#
#    _queue.put(None) # signal end of queue
#
#    _queue.close()  # indicate this local thread is done
#    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore
#
##
## masked langauge model sequence loader from multiple files 
##
#def multiprocess_mlm_sequence_loader(epoch: int, process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,sequence_type,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):
#
#    torch.manual_seed(_config["random_seed"] + 10000 * epoch)
#    numpy.random.seed(_config["random_seed"] + 10000 * epoch)
#    random.seed(_config["random_seed"] + 10000 * epoch)
#
#    if _config["token_embedder_type"] == "bert_cat":
#        _tokenizer = BlingFireTokenizer()
#        _ind = PretrainedTransformerIndexer(pretrained_model=_config["bert_pretrained_model"], do_lowercase=True)
#        _token_indexers = {"tokens": _ind}
#
#        _tuple_loader = IrSingleSequenceDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                max_seq_length= _config["max_doc_length"], min_seq_length=_config["min_doc_length"],)
#
#    else:
#        _tokenizer = BlingFireTokenizer()
#
#        if _config["token_embedder_type"] == "embedding":
#            _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
#            _vocab = Vocabulary.from_files(_config["vocab_directory"])
#
#        elif _config["token_embedder_type"] == "fasttext":
#            _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
#            _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])
#
#        elif _config["token_embedder_type"] == "elmo":
#            _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
#            _vocab = None
#        elif _config["token_embedder_type"] == "bert_embedding" or _config["token_embedder_type"] == "bert_vectors" or _config["token_embedder_type"] == "bert_dot":
#            _tokenizer = PretrainedTransformerTokenizer(_config["bert_pretrained_model"])               
#            _ind = PretrainedTransformerIndexer(model_name=_config["bert_pretrained_model"])            
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#        elif _config["token_embedder_type"] == "huggingface_bpe":
#
#            files = _config["bpe_vocab_files"].split(";")
#
#            _ind = CustomTransformerIndexer(CharBPETokenizer(files[0],files[1], lowercase=True))
#            _tokenizer = _ind._allennlp_tokenizer
#            _token_indexers = {"tokens": _ind}
#            _vocab = Vocabulary()
#
#        _tuple_loader = MLMMaskedSequenceDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, 
#                                                    max_doc_length= _config["max_doc_length"], min_doc_length=_config["min_doc_length"],
#                                                    mask_probability=_config["mlm_mask_probability"],mlm_mask_replace_probability=_config["mlm_mask_replace_probability"],
#                                                    mlm_mask_whole_words =_config["mlm_mask_whole_words"],mlm_mask_random_probability = _config["mlm_mask_random_probability"],
#                                                    bias_sampling_method=_config["mlm_bias_sampling_method"],bias_merge_alpha=_config["mlm_bias_merge_alpha"])
#
#    data_reader = _tuple_loader.read(_local_file)
#    data_reader.index_with(_vocab)
#    loader = SimpleDataLoader(data_reader, batch_size=int(_config["batch_size_pretrain"]))
#
#    for training_batch in loader:
#
#        _queue.put(training_batch)  # this moves the tensors in to shared memory
#
#    _queue.put(None) # signal end of queue
#
#    _queue.close()  # indicate this local thread is done
#    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore