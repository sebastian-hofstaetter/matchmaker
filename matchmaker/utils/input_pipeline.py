import torch
import numpy
import random

from allennlp.data.samplers import BucketBatchSampler, MaxTokensBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader

from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


from matchmaker.dataloaders.concatenated_reranking_loader import *
from matchmaker.dataloaders.concatenated_training_loader import *

from matchmaker.dataloaders.independent_reranking_loader import *
from matchmaker.dataloaders.independent_training_loader import *

from matchmaker.dataloaders.id_sequence_loader import *
from matchmaker.dataloaders.mlm_masked_sequence_loader import *
from matchmaker.dataloaders.tas_balanced_training_loader import *
from transformers import AutoTokenizer

from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer
from matchmaker.dataloaders.transformer_tokenizer import FastTransformerTokenizer
from matchmaker.modules.bert_embedding_token_embedder import PretrainedBertIndexerNoSpecialTokens


from typing import Dict, Tuple, List
#from tokenizers import ByteLevelBPETokenizer,CharBPETokenizer
#from matchmaker.dataloaders.transformer_tokenizer import CustomTransformerTokenizer,CustomTransformerIndexer

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system") # VERY MUCH needed for linux !! makes everything faster, but tends to break stuff


def allennlp_single_sequence_loader(model_config, run_config, _input_file, sequence_type, force_exact_batch_size=False):
    '''
    Load examples from a .tsv file in the single sequence format: id<tab>text

    (Using allennlp's v2 multiprocess loader)
    '''
    if sequence_type == "query":
        max_length = model_config["max_query_length"]
        min_length = model_config["min_query_length"]
        batch_size = run_config["query_batch_size"]
    else:  # doc
        max_length = model_config["max_doc_length"]
        min_length = model_config["min_doc_length"]
        batch_size = run_config["collection_batch_size"]

    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max_length)

    reader = IdSequenceDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                           max_seq_length=max_length, min_seq_length=min_length, sequence_type=sequence_type)

    if force_exact_batch_size:
        loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(batch_size)*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_size=int(batch_size))
    else:
        loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(batch_size)*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_sampler=MaxTokensBatchSampler(max_tokens=int(batch_size)*max_length, sorting_keys=["seq_tokens"], padding_noise=0))
    loader.index_with(_vocab)
    return loader


def allennlp_triple_training_loader(model_config, run_config, _input_file):
    '''
    Load training examples (either in the re-ranking text file format or a dynamic loader)

    (Using allennlp's v2 multiprocess loader)
    '''
    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max(run_config["max_doc_length"], run_config["max_query_length"]))

    if run_config.get("dynamic_sampler", False) == False:

        if model_config.get("model_input_type", "") == "concatenated" or model_config["token_embedder_type"] == "bert_cat":

            reader = ConcatenatedTrainingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                             max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                             min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                             data_augment=run_config["train_data_augment"], train_pairwise_distillation=run_config["train_pairwise_distillation"], train_qa_spans=run_config["train_qa_spans"])
        else:
            reader = IndependentTrainingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                           max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                           min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                           data_augment=run_config["train_data_augment"], train_pairwise_distillation=run_config["train_pairwise_distillation"],
                                           query_augment_mask_number=run_config["query_augment_mask_number"], train_qa_spans=run_config["train_qa_spans"])

        loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(run_config["batch_size_train"])*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_size=run_config["batch_size_train"])
        loader.index_with(_vocab)

    else:
        #if run_config["dynamic_sampler_type"] == "list":
        #    loader = IrDynamicTripleDatasetLoader(query_file=run_config["dynamic_query_file"], collection_file=run_config["dynamic_collection_file"],
        #                                          qrels_file=run_config["dynamic_qrels_file"], candidate_file=run_config["dynamic_candidate_file"],
        #                                          batch_size=int(run_config["batch_size_train"]), queries_per_batch=run_config["dynamic_queries_per_batch"], tokenizer=_tokenizer, token_indexers=_token_indexers,
        #                                          max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
        #                                          min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
        #                                          data_augment=run_config["train_data_augment"], vocab=_vocab)

        if run_config["dynamic_sampler_type"] == "tas_balanced":
            loader = TASBalancedDatasetLoader(query_file=run_config["dynamic_query_file"], collection_file=run_config["dynamic_collection_file"],
                                              pairs_with_teacher_scores=run_config["dynamic_pairs_with_teacher_scores"], query_cluster_file=run_config["dynamic_query_cluster_file"],
                                              batch_size=int(run_config["batch_size_train"]), clusters_per_batch=run_config["dynamic_clusters_per_batch"], tokenizer=_tokenizer,
                                              max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                              pair_balancing_strategy=run_config["tas_balanced_pair_strategy"],random_seed =run_config["random_seed"])
        else:
            raise ConfigurationError("dynamic sampler type not supported")

    return loader


def allennlp_reranking_inference_loader(model_config, run_config, _input_file):
    '''
    Load examples from a .tsv file in the reranking candidate file format: q_id<tab>d_id<tab>q_text<tab>d_text

    (Using allennlp's v2 multiprocess loader)
    '''

    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max(run_config["max_doc_length"], run_config["max_query_length"]))

    if model_config.get("model_input_type", "") == "concatenated" or model_config["token_embedder_type"] == "bert_cat":

        reader = ConcatenatedReRankingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                                    max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                                    min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                                    train_qa_spans=run_config["train_qa_spans"])
    else:

        reader = IndependentReRankingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                                   max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                                   min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                                   query_augment_mask_number=run_config["query_augment_mask_number"], train_qa_spans=run_config["train_qa_spans"])

    loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                    max_instances_in_memory=int(run_config["batch_size_eval"])*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                    batch_sampler=MaxTokensBatchSampler(max_tokens=int(run_config["batch_size_eval"])*run_config["max_doc_length"], sorting_keys=["doc_tokens"], padding_noise=0))
    loader.index_with(_vocab)
    return loader


# def allennlp_mlm_loader(model_config, run_config, _input_file):


def _get_indexer(model_config, max_length):
    # default values
    _tokenizer = BlingFireTokenizer()
    _vocab = Vocabulary()

    if model_config["token_embedder_type"] == "embedding":
        _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        _vocab = Vocabulary.from_files(model_config["vocab_directory"])

    elif model_config["token_embedder_type"] == "bert_embedding" or model_config["token_embedder_type"] == "bert_vectors":
        _tokenizer = PretrainedTransformerTokenizer(model_config["bert_pretrained_model"], do_lowercase=True, start_tokens=[], end_tokens=[])
        _ind = PretrainedBertIndexerNoSpecialTokens(pretrained_model=model_config["bert_pretrained_model"], do_lowercase=True, max_pieces=max_length)
        _token_indexers = {"tokens": _ind}

    elif model_config["token_embedder_type"].startswith("bert"):
        model = model_config["bert_pretrained_model"]
        if "facebook/dpr" in model:
            model = "bert-base-uncased"     # should be the right one (judging from paper + huggingface doc)
        _tokenizer = FastTransformerTokenizer(model)
        _token_indexers = None

    return _tokenizer, _token_indexers, _vocab
