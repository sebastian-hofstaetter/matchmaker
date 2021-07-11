import os
import copy
import time
import glob
from typing import Dict, Tuple, List
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

from matchmaker.utils.core_metrics import *

from matchmaker.utils.cross_experiment_cache import *
from matchmaker.utils.input_pipeline import *
from matchmaker.utils.performance_monitor import *
import time
#
# run eval of a neural-ir model
# -------------------------------
#
# validate_model(...) = during training validation with parameter searches (saves all info in csv files)
# test_model(...) = after training do only inference on fixed params 

evaluate_cache={}

#
# raw model evaluation, returns model results as python dict, does not save anything / no metrics
#
def evaluate_model(model, config, _logger, cuda_device, eval_tsv, use_cache=False, 
                   output_secondary_output: bool = False):

    model.eval()  # turning off training
    validation_results = {}
    qa_validation_results_stats = defaultdict(list)
    qa_validation_results = defaultdict(list)
    secondary_output = {}
    secondary_output["per_batch_info"] = {}
    fill_cache=False
    cached_batches = None

    torch.cuda.empty_cache()

    concated_sequences = False
    if config["token_embedder_type"] == "bert_cat":
        concated_sequences = True

    use_title_body_sep = config["use_title_body_sep"]
    use_cls_scoring = config["use_cls_scoring"]
    train_sparsity = config["minimize_sparsity"]
    train_qa_spans = config["train_qa_spans"]
    use_fp16 = config["use_fp16"]

    use_submodel_caching = False
    batch_latency = []
    try:
        if use_cache:
            global evaluate_cache
            if eval_tsv not in evaluate_cache:
                fill_cache=True
                evaluate_cache[eval_tsv] = []
            cached_batches = evaluate_cache[eval_tsv]
        use_submodel_caching = "submodel_validation_cache_path" in config
        if use_submodel_caching:
            submodel_cacher = CrossExperimentReplayCache(os.path.join(config["submodel_validation_cache_path"],os.path.basename(os.path.dirname(eval_tsv))),is_readonly=config["submodel_validation_cache_readonly"])

        if not use_cache or fill_cache:
            input_loader = allennlp_reranking_inference_loader(config, config, eval_tsv)
            _logger.info("[eval_model] --- Start validation from loader")
        else:
            input_loader = cached_batches
            _logger.info("[eval_model] --- Start validation with cache size:" + str(len(cached_batches)))

        with torch.no_grad():
            pair_count = 0
            if train_sparsity:
                stopwords=0
                total_words=0
            i=0
            for batch_orig in input_loader:
                with torch.cuda.amp.autocast(enabled=config["use_fp16"]):

                    if fill_cache:
                        cached_batches.append(batch_orig)

                    start_time = time.time()
                    batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)

                    #
                    # create input 
                    #
                    args_in = []
                    if concated_sequences:
                        args_in.append(batch["doc_tokens"])  
                    else:
                        args_in += [batch["query_tokens"],batch["doc_tokens"]]

                    if use_title_body_sep:
                        args_in.append(batch["title_tokens"])
                    
                    if use_submodel_caching:
                        cached_parts = submodel_cacher.get_next()
                        if cached_parts != None: cached_parts = move_to_device(cached_parts,cuda_device)
                        args_in.append(cached_parts)

                    output = model.forward(*args_in, output_secondary_output=output_secondary_output, use_fp16 = use_fp16)

                    #if concated_sequences:
                    #    torch.cuda.empty_cache()
                    #    output = model.forward(batch["doc_tokens"],output_secondary_output=output_secondary_output)
                    #elif use_title_body_sep and use_cls_scoring:
                    #    output,_,_ = model.forward(batch["query_tokens"], batch["doc_tokens"], batch["title_tokens"],output_secondary_output)
                    #elif use_title_body_sep:
                    #    output = model.forward(batch["query_tokens"], batch["doc_tokens"], batch["title_tokens"],output_secondary_output)
                    #elif use_submodel_caching:
                    #    output = model.forward(batch["query_tokens"], batch["doc_tokens"],output_secondary_output,cached_parts)
                    #else:
                    #    output = model.forward(batch["query_tokens"], batch["doc_tokens"],output_secondary_output=output_secondary_output)
    
                    if output_secondary_output:
                        if train_sparsity:
                            #output,cache_parts_out, secondary, sparsity_vec,sparsity_stats = output
                            output,sparsity_vec = output
                        elif train_qa_spans:
                            output, secondary, answerability,qa_logits_start,qa_logits_end = output 
                            #answerability = answerability.cpu().float()
                            #qa_logits_start = qa_logits_start.cpu().float()
                            #qa_logits_end = qa_logits_end.cpu().float()
                        else:
                            output, secondary = output
                        secondary_cpu = {}
                        for tens_name,tens in secondary.items():
                            if type(tens) is torch.Tensor:
                                secondary_cpu[tens_name] = tens.cpu()
                            else:
                                if tens_name not in secondary_output["per_batch_info"]:
                                    secondary_output["per_batch_info"][tens_name] = []
                                secondary_output["per_batch_info"][tens_name].append(tens)
                    else:
                        if train_qa_spans:
                            output,answerability,qa_logits_start,qa_logits_end = output 
                            #answerability = answerability.cpu().float()
                            #qa_logits_start = qa_logits_start.cpu().float()
                            #qa_logits_end = qa_logits_end.cpu().float()

                        if train_sparsity:
                            #output, cache_parts_out, sparsity_vec,sparsity_stats = output
                            output, sparsity_vec = output

                    if use_submodel_caching and cached_parts == None: 
                        submodel_cacher.cache(cache_parts_out)


                    #if train_sparsity:
                    #    ctw = batch["doc_tokens"]["tokens"]["tokens"]!=0
                    #    stopwords += float(((sparsity_vec <= 0) * ctw.unsqueeze(1)).float().sum())
                    #    total_words += float(ctw.float().sum())

                    output = output.cpu()  # get the output back to the cpu - in one piece
                    #_logger.info("val_output" + str(i))

                    if train_qa_spans:
                        start_idx = torch.max(qa_logits_start,dim=-1).indices.cpu()
                        end_idx = torch.max(qa_logits_end,dim=-1).indices.cpu()
                        answerable = (torch.max(answerability,dim=-1).indices == 1).cpu()


                    for sample_i, sample_query_id in enumerate(batch_orig["query_id"]):  # operate on cpu memory
                        sample_doc_id = batch_orig["doc_id"][sample_i]  # again operate on cpu memory
                        if sample_query_id not in validation_results:
                            validation_results[sample_query_id] = []
                            qa_validation_results[sample_query_id] = {}
                            if output_secondary_output:
                                secondary_output[sample_query_id] = {}
                    
                        if train_qa_spans:
                            if batch_orig["qa_hasAnswer"][sample_i] == 1:
                                qa_validation_results_stats["QA/Start_accuracy"].append(1 if start_idx[sample_i] in batch_orig["qa_start"][sample_i] else 0)
                                qa_validation_results_stats["QA/End_accuracy"].append(1 if end_idx[sample_i] in batch_orig["qa_end"][sample_i] else 0)
                                qa_validation_results_stats["QA/Full_accuracy"].append(1 if end_idx[sample_i] in batch_orig["qa_end"][sample_i] and start_idx[sample_i] in batch_orig["qa_start"][sample_i] else 0)
                            
                            qa_validation_results_stats["QA/Answerability_accuracy"].append(float(answerable[sample_i] == batch_orig["qa_hasAnswer"][sample_i]))
                            if answerable[sample_i]:
                                os=batch_orig["offsets_start"][sample_i][start_idx[sample_i]]
                                oe=batch_orig["offsets_end"][sample_i][end_idx[sample_i]]
                                if oe != None and os != None and oe >= os:
                                    qa_validation_results[sample_query_id][sample_doc_id] = (bool(answerable[sample_i]), int(start_idx[sample_i]), int(end_idx[sample_i]), batch_orig["doc_text"][sample_i][os:oe])
                                else: 
                                    qa_validation_results[sample_query_id][sample_doc_id] = (False)
                            else:
                                qa_validation_results[sample_query_id][sample_doc_id] = (False)
    
                        pair_count +=1
                        validation_results[sample_query_id].append((sample_doc_id, float(output[sample_i])))
    
                        if output_secondary_output:
                            secondary_i = {}
                            for tens_name,tens in secondary_cpu.items():
                                secondary_i[tens_name] = tens[sample_i].data.numpy()
                            secondary_output[sample_query_id][sample_doc_id] = secondary_i
                    batch_latency.append((time.time()-start_time)*1000)
                i+=1

            #if train_sparsity and total_words > 0:
            #    with open(config["sparsity_log_path"], "a") as log_file:
            #        log_file.write(str(eval_tsv) + "\t" +str(stopwords) + "\t" + str(total_words) + "\t" + str(stopwords/total_words) +"\n")
            #_logger.info("val_close")

        #perf_monitor = PerformanceMonitor.get()
        #perf_monitor.log_unique_value("eval_batch_latency_ms", batch_latency)

        if use_submodel_caching:
            submodel_cacher.save(True)

    except BaseException as e:
        _logger.info('-' * 89)
        _logger.exception('[eval_model] Got exception: ')
        print("----- Attention! - something went wrong in eval_model (see logger) ----- ")
        
        raise e
    
    for k,v in qa_validation_results_stats.items():
        if len(v) == 0:
            qa_validation_results_stats[k] = 0
        else:
            qa_validation_results_stats[k] = sum(v) / float(len(v))

    return validation_results,qa_validation_results,qa_validation_results_stats,pair_count,secondary_output

#
# validate a model during training + save results, metrics 
#
# validation_config = { 
#   tsv,
#   _candidate_set_from_to
#   _qrels
# }
#
#
def validate_model(val_name, model, config, validation_config, run_folder, _logger, cuda_device, 
                   epoch_number=-1, batch_number=-1, 
                   global_best_info=None, candidate_set = None, use_cache=False, 
                   output_secondary_output: bool = False,is_distributed=False):

    if val_name == "cont":
        evaluation_id = str(epoch_number) + "-" +str(batch_number)
    else:
        evaluation_id = "end-" + val_name

    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("validation")

    validation_results,qa_validation_results,qa_validation_results_stats,pair_count,secondary_output = evaluate_model(model,config,_logger,cuda_device,validation_config["tsv"],use_cache,output_secondary_output)
    
    perf_monitor.stop_block("validation",pair_count)

    #_logger.info("val_before_unrolled")
    if val_name != "cont":
        validation_file_path = os.path.join(run_folder, "val-"+evaluation_id+"-output.txt")
        save_sorted_results(validation_results, validation_file_path)

    ranked_results = unrolled_to_ranked_result(validation_results)

    #
    # write out secondary output
    #
    if output_secondary_output:
        m=model
        if is_distributed:
            m=model.module
        save_secondary_output(m,os.path.join(run_folder, "secondary-"+evaluation_id+".npz"),ranked_results,secondary_output,config["secondary_output"]["top_n"])

    #
    # compute ir metrics (for ms_marco) and output them (to the logger + own csv file)
    # ---------------------------------
    #
    best_metric_info = {}
    best_metric_info["epoch"] = epoch_number
    best_metric_info["batch_number"] = batch_number
  
    #
    # do a cs@n over multiple n evaluation
    #
    if candidate_set != None:

        metrics = calculate_metrics_along_candidate_depth(ranked_results,load_qrels(validation_config["qrels"]),candidate_set,validation_config["candidate_set_from_to"],validation_config["binarization_point"])
        #_logger.info("val_after_metrics")

        #metrics = compute_metrics_with_cs_at_n(validation_config["qrels"],validation_file_path,candidate_set,validation_config["candidate_set_from_to"])

        #perf_monitor.stop_block("old_metrics",1)

        # save main validation metric overview
        metric_file_path = os.path.join(run_folder, "validation-main-all.csv")
        save_one_metric_multiN(metric_file_path,metrics,
                               config["validation_metric"],range(validation_config["candidate_set_from_to"][0],validation_config["candidate_set_from_to"][1] + 1),
                               epoch_number,batch_number)

        # save all info + get best value of validation metric
        best_metric_value = 0
        for current_cs_n in range(validation_config["candidate_set_from_to"][0],validation_config["candidate_set_from_to"][1] + 1):
            metric_file_path = os.path.join(run_folder, "validation-metrics-cs_"+str(current_cs_n)+".csv")
            if val_name == "cont":
                save_fullmetrics_oneN(metric_file_path,metrics[current_cs_n],epoch_number,batch_number)
            if metrics[current_cs_n][config["validation_metric"]] > best_metric_value:
                best_metric_value = metrics[current_cs_n][config["validation_metric"]]
                best_metric_info["metrics"] = metrics[current_cs_n]
                best_metric_info["cs@n"] = current_cs_n

        # save at the end all in one file
        if val_name != "cont":
            save_fullmetrics_rangeN(os.path.join(run_folder, "val-"+evaluation_id+"-all-metrics.csv"),metrics,range(validation_config["candidate_set_from_to"][0],validation_config["candidate_set_from_to"][1] + 1))

    #
    # do a 1x evaluation over the full given validation set
    #
    else:
        metrics = calculate_metrics_plain(ranked_results,load_qrels(validation_config["qrels"]),validation_config["binarization_point"])
        metric_file_path = os.path.join(run_folder, "validation-metrics.csv")
        save_fullmetrics_oneN(metric_file_path,metrics,epoch_number,batch_number)
        best_metric_info["metrics"] = metrics
        best_metric_info["cs@n"] = "-"
        best_metric_value = metrics[config["validation_metric"]]

    #
    # save best results
    #
    if val_name == "cont":
        if validation_config["save_only_best"] == True and global_best_info != None:
            #os.remove(validation_file_path)
            if best_metric_value > global_best_info["metrics"][config["validation_metric"]]:
                validation_file_path = os.path.join(run_folder, "best-validation-output.txt")
                save_sorted_results(validation_results, validation_file_path)
    #else:
    #    validation_file_path = os.path.join(run_folder, "val-"+evaluation_id+"-output.txt")
    #    save_sorted_results(validation_results, validation_file_path)

    #
    # qa eval 
    #
    if config["train_qa_spans"] and "qa_answers" in validation_config:
        answers = validation_config["qa_answers"]
        qid_answers = {}
        with open(answers, "r",encoding="utf8") as answer_f:
            for l in answer_f:
                l=l.split("\t")
                l[-1] = l[-1].rstrip()
                qid_answers[l[0]] = l[1:]
        
        tokenizer = PretrainedTransformerTokenizer(config["bert_pretrained_model"])

        model_answers={}
        exact_scores={}
        f1_scores={}
        for q_id,doc_ids in ranked_results.items():
            model_answers[q_id] = ""
            for doc in doc_ids:
                if qa_validation_results[q_id][doc] != False:
                    model_answers[q_id] = qa_validation_results[q_id][doc][3]
                    break
            if q_id in qid_answers:
                exact_scores[q_id] = max(compute_exact(a, model_answers[q_id]) for a in qid_answers[q_id])
                f1_scores[q_id] = max(compute_f1(a, model_answers[q_id]) for a in qid_answers[q_id])
            #else:
            #    exact_scores[q_id] = float(qa_validation_results[q_id][doc][0] == False)
            #    f1_scores[q_id] = float(qa_validation_results[q_id][doc][0] == False)
        qa_validation_results_stats["QA/ExactMatch_TopRanked"] = sum(exact_scores.values()) / len(exact_scores.values())
        qa_validation_results_stats["QA/F1_Term_Overlap_TopRanked"] = sum(f1_scores.values()) / len(f1_scores.values())

        save_qa_answers(model_answers, qid_answers, os.path.join(run_folder, "last-qa-output.tsv"))

    return best_metric_info, metrics, pair_count,qa_validation_results_stats

#
# test a model after training + save results, metrics 
#
def test_model(model, config,test_config, run_folder, _logger, cuda_device, test_name,candidate_set = None, candidate_set_n = None,output_secondary_output=False,is_distributed=False):

    test_results,qa_validation_results,qa_validation_results_stats,_,secondary_output = evaluate_model(model,config,_logger,cuda_device,test_config["tsv"],use_cache=False,output_secondary_output=output_secondary_output)

    #
    # save sorted results
    #
    test_file_path = os.path.join(run_folder, test_name+"-output.txt")
    save_sorted_results(test_results, test_file_path) 

    #
    # compute ir metrics (for ms_marco) and output them (to the logger + own csv file)
    # ---------------------------------
    #
    metrics = None
    if "qrels" in test_config:
        ranked_results = unrolled_to_ranked_result(test_results)

        if candidate_set != None:
            metrics = calculate_metrics_single_candidate_threshold(ranked_results,load_qrels(test_config["qrels"]),candidate_set,candidate_set_n,test_config["binarization_point"])

        else:
            metrics = calculate_metrics_plain(ranked_results,load_qrels(test_config["qrels"]),test_config["binarization_point"])

        metric_file_path = os.path.join(run_folder, test_name+"-metrics.csv")
        save_fullmetrics_oneN(metric_file_path, metrics, -1, -1)

    if output_secondary_output:
        ranked_results = unrolled_to_ranked_result(test_results)
        m=model
        if is_distributed:
            m=model.module
        save_secondary_output(m,os.path.join(run_folder, "secondary-"+test_name+".npz"),ranked_results,secondary_output,config["secondary_output"]["top_n"])

    return metrics

def save_secondary_output(model,out_file,ranked_results,secondary_output,max_sec_i):
    filtered_secondary_output = {}
    for q_id,ranked_doc_ids in ranked_results.items():
        filtered_secondary_output[q_id] = {}
        for i,doc_id in enumerate(ranked_doc_ids):
            if i == max_sec_i:
                break
            filtered_secondary_output[q_id][doc_id] = secondary_output[q_id][doc_id]
    model_data_secondary = model.get_param_secondary()
    model_data_secondary_cpu={}
    for tens_name,tens in model_data_secondary.items():
        model_data_secondary_cpu[tens_name] = tens.data.cpu().numpy()

    numpy.savez_compressed(out_file,model_data=model_data_secondary_cpu, qd_data=filtered_secondary_output,per_batch_info=secondary_output["per_batch_info"])


def save_qa_answers(results, labels, file):
    with open(file, "w") as val_file:
        for query_id, query_data in results.items():
            if query_id in labels:
                val_file.write("\t".join(str(x) for x in [query_id,query_data]+labels[query_id])+"\n")
            else:
                val_file.write("\t".join(str(x) for x in [query_id,query_data,"<no answ.>"])+"\n")

def save_sorted_results(results, file, until_rank=-1):
    with open(file, "w") as val_file:
        lines = 0
        for query_id, query_data in results.items():
            
            # sort the results per query based on the output
            for rank_i, (doc_id, output_value) in enumerate(sorted(query_data, key=lambda x: x[1], reverse=True)):
                val_file.write("\t".join(str(x) for x in [query_id, doc_id, rank_i + 1, output_value])+"\n")
                lines += 1
                if until_rank > -1 and rank_i == until_rank + 1:
                    break
    return lines

def save_fullmetrics_oneN(file, metrics, epoch_number, batch_number):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\nEpoch,After_Batch," + ",".join(k for k, v in metrics.items())+"\n")
    # append single row
    with open(file, "a") as metric_file:
        metric_file.write(str(epoch_number) + "," +str(batch_number) + "," + ",".join(str(v) for k, v in metrics.items())+"\n")

def save_fullmetrics_rangeN(file, metrics, m_range):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\ncs@n," + ",".join(k for k, v in metrics[m_range.start].items())+"\n")
    # append single row
    with open(file, "a") as metric_file:
        for cs_n in m_range:
            metric_file.write(str(cs_n) + "," + ",".join(str(v) for k, v in metrics[cs_n].items())+"\n")


def save_best_info(file, best_info):
    with open(file, "w") as metric_file:
        metric_file.write("sep=,\nEpoch,batch_number,cs@n," + ",".join(k for k, v in best_info["metrics"].items())+"\n")
        metric_file.write(str(best_info["epoch"]) + "," +str(best_info["batch_number"])+ "," +str(best_info["cs@n"]) + "," + ",".join(str(v) for k, v in best_info["metrics"].items())+"\n")


def save_one_metric_multiN(file, metrics, selected_metric, over_range, epoch_number, batch_number):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\nEpoch,After_Batch," + ",".join(str(k) for k in over_range)+"\n")

    # append single row
    with open(file, "a") as metric_file:
        metric_file.write(str(epoch_number) + "," +str(batch_number) + "," + ",".join(str(v[selected_metric]) for cs_at_n, v in metrics.items())+"\n")
