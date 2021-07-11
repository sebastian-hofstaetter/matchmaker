import time
from typing import Any, Dict, Iterator, List
import copy

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import traceback
from allennlp.data.data_loaders.multiprocess_data_loader import WorkerError

from allennlp.nn.util import move_to_device
from matchmaker.utils.config import *
from matchmaker.models.all import get_model, get_word_embedder, build_model
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from transformers.file_utils import cached_path,hf_bucket_url,WEIGHTS_NAME

mp.set_sharing_strategy("file_system") # VERY MUCH needed for linux !! makes everything MUCH faster

from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index

from rich.console import Console

def data_parallel_prepare(module, device_ids=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    #if not isinstance(inputs, tuple):
    #    inputs = (inputs,)

    #device_type = _get_available_device_type()

    #if device_ids is None:
    #    device_ids = _get_all_device_indices()

    #if output_device is None:
    #    output_device = device_ids[0]

    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    #output_device = _get_device_index(output_device, True)
    #src_device_obj = torch.device(device_type, device_ids[0])

    #for t in chain(module.parameters(), module.buffers()):
    #    if t.device != src_device_obj:
    #        raise RuntimeError("module must have its parameters and buffers "
    #                           "on device {} (device_ids[0]) but found one of "
    #                           "them on device: {}".format(src_device_obj, t.device))

    #inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    #if len(device_ids) == 1:
    #    return module(*inputs[0], **module_kwargs[0])
    #used_device_ids = #device_ids[:len(inputs)]
    replicas = replicate(module, device_ids)
    #outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return replicas

def data_parallel_forward(replicas, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    #device_type = _get_available_device_type()

    #if device_ids is None:
    #    device_ids = _get_all_device_indices()

    if output_device is None:
        output_device = device_ids[0]

    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    output_device = _get_device_index(output_device, True)
    #src_device_obj = torch.device(device_type, device_ids[0])

    #for t in chain(module.parameters(), module.buffers()):
    #    if t.device != src_device_obj:
    #        raise RuntimeError("module must have its parameters and buffers "
    #                           "on device {} (device_ids[0]) but found one of "
    #                           "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    #if len(device_ids) == 1:
    #    return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    #replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

class DynamicTeacher():
    '''
    Wraps a trained model checkpoint and the training batch queue to score (inference only) samples from the batch
    '''

    def __init__(self,
                 config: Dict[str,Any],
                 dataloader:DataLoader,
                 logger):

        super().__init__()
        self.config = config
        self.dynamic_teacher_path = config["dynamic_teacher_path"]
        self.dynamic_teacher_in_batch_scoring = config["dynamic_teacher_in_batch_scoring"]
        self.dynamic_teacher_per_term_scores = config.get("dynamic_teacher_per_term_scores",False)

        self.wrapped_dataloader = dataloader

        self.cuda_device = torch.cuda.device_count() - 1 # [torch.cuda.device_count() - 2,torch.cuda.device_count() - 1] # take the last gpu 
        self.logger = logger

    def __iter__(self) -> Iterator[TensorDict]:

        ctx = mp.get_context("spawn") # need spawn here, otherwise CUDA fails 

        queue: mp.JoinableQueue = ctx.JoinableQueue()
        worker = ctx.Process(
            target=self.dynamic_teacher_subprocess, args=(queue,), #daemon=True
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


    def dynamic_teacher_subprocess(self, queue):
        
        try:
            console = Console()

            console.log("[DynamicTeacher] Load teacher model from: " + self.dynamic_teacher_path)

            #
            # load model
            #
            model_config = get_config_single(self.dynamic_teacher_path)
            word_embedder, padding_idx = get_word_embedder(model_config)
            model, encoder_type = get_model(model_config,word_embedder,padding_idx)
            model = build_model(model,encoder_type,word_embedder,model_config)
            model.is_teacher_model = True
            if model_config.get("model_checkpoint_from_huggingface",False):
                model_path = cached_path(hf_bucket_url(self.dynamic_teacher_path, WEIGHTS_NAME))
            else:
                model_path = os.path.join(self.dynamic_teacher_path, "best-model.pytorch-state-dict")
            load_result = model.load_state_dict(torch.load(model_path,map_location="cpu"),strict=False)
            self.logger.info('[DynamicTeacher] Warmstart init model from:  %s', model_path)
            self.logger.info(load_result)
            console.log("[DynamicTeacher] Warmstart Result:",load_result)
            model = model.eval()

            use_multi = False
            if type(self.cuda_device) == int:
                model = model.cuda(self.cuda_device)

            #
            # multi gpu 
            #
            else:
                use_multi = True
                model = model.cuda(self.cuda_device[0])
                replicas = data_parallel_prepare(model,self.cuda_device)

            use_fp16 = model_config["use_fp16"]
            concated_sequences = False
            if model_config["token_embedder_type"] == "bert_cat":
                concated_sequences = True
            #use_title_body_sep = model_config["use_title_body_sep"]
            #train_sparsity = model_config["minimize_sparsity"]
            #train_qa_spans = model_config["train_qa_spans"]

            #
            # connect to pipeline
            #
            console.log("[DynamicTeacher] Run Teacher Inference ...")

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                for orig_batch in self.wrapped_dataloader:

                    if use_multi:
                        batch = move_to_device(copy.deepcopy(orig_batch), self.cuda_device[0])
                        batch_neg = move_to_device(copy.deepcopy(orig_batch), self.cuda_device[1])
                    else:
                        batch = move_to_device(copy.deepcopy(orig_batch), self.cuda_device)
                        batch_neg = batch

                    pos_in = []
                    neg_in = []
                    if concated_sequences:
                        pos_in.append(batch["doc_pos_tokens"])  
                        neg_in.append(batch_neg["doc_neg_tokens"])
                    else:
                        pos_in += [batch["query_tokens"],batch["doc_pos_tokens"]]
                        neg_in += [batch_neg["query_tokens"],batch_neg["doc_neg_tokens"]]

                    #if use_title_body_sep:
                    #    pos_in.append(batch["title_pos_tokens"])
                    #    neg_in.append(batch_neg["title_neg_tokens"])

                    #if train_qa_spans: # add start positions for qa training (used to anchor end logits on the start ground truth)
                    #    pos_in.append(batch["pos_qa_start"])

                    #
                    # run model forward
                    #
                    if use_multi:
                        output_pos, output_neg = parallel_apply(replicas, [pos_in,neg_in], [{"use_fp16": use_fp16},{"use_fp16": use_fp16}], self.cuda_device)
                        #output_neg = data_parallel_forward(replicas, *neg_in, device_ids=cuda_device, use_fp16 = use_fp16)
                        output_pos, query_vecs_pos, doc_vecs_pos = output_pos
                        output_neg, query_vecs_neg, doc_vecs_neg = output_neg
                            # colbert model
                        ib_output_pos = model.forward_inbatch_aggregation(query_vecs_pos,batch["query_tokens"]["attention_mask"], doc_vecs_pos, batch["doc_pos_tokens"]["attention_mask"])
                        ib_output_neg = model.forward_inbatch_aggregation(query_vecs_neg,batch["query_tokens"]["attention_mask"], doc_vecs_neg, batch["doc_neg_tokens"]["attention_mask"])
                        orig_batch["dyn_teacher_scores_pos"] = ib_output_pos.cpu()
                        orig_batch["dyn_teacher_scores_neg"] = ib_output_neg.cpu()

                    else:
                        output_pos = model.forward(*pos_in, use_fp16 = use_fp16)
                        output_neg = model.forward(*neg_in, use_fp16 = use_fp16)

                    #if train_qa_spans:
                    #    output,answerability,qa_logits_start,qa_logits_end = output 
                        #answerability = answerability.cpu().float()
                        #qa_logits_start = qa_logits_start.cpu().float()
                        #qa_logits_end = qa_logits_end.cpu().float()

                    #if train_sparsity:
                    #    output, cache_parts_out, sparsity_vec,sparsity_stats = output
                        if self.dynamic_teacher_per_term_scores:
                            (*output_pos, per_term_scores_pos) = output_pos
                            (*output_neg, per_term_scores_neg) = output_neg

                            orig_batch["dyn_teacher_per_term_scores_pos"] = per_term_scores_pos.cpu()
                            orig_batch["dyn_teacher_per_term_scores_neg"] = per_term_scores_neg.cpu()

                        if self.dynamic_teacher_in_batch_scoring:

                            output_pos, query_vecs_pos, doc_vecs_pos = output_pos
                            output_neg, query_vecs_neg, doc_vecs_neg = output_neg

                            # colbert model
                            ib_output_pos = model.forward_inbatch_aggregation(query_vecs_pos,batch["query_tokens"]["attention_mask"], doc_vecs_pos, batch["doc_pos_tokens"]["attention_mask"])
                            ib_output_neg = model.forward_inbatch_aggregation(query_vecs_neg,batch["query_tokens"]["attention_mask"], doc_vecs_neg, batch["doc_neg_tokens"]["attention_mask"])

                            orig_batch["dyn_teacher_scores_pos"] = ib_output_pos.cpu()
                            orig_batch["dyn_teacher_scores_neg"] = ib_output_neg.cpu()

                        else:
                            orig_batch["dyn_teacher_scores_pos"] = output_pos.cpu()
                            orig_batch["dyn_teacher_scores_neg"] = output_neg.cpu()

                    queue.put((orig_batch,None))  # this moves the tensors in to shared memory

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))
        
        queue.put((None, None))
        # Wait until this process can safely exit.
        queue.join()