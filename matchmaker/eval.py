import os
import copy
import time
import glob
from typing import Dict, Tuple, List

import torch
import numpy

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

from matchmaker.core_metrics import *

#from evaluation.msmarco_eval import *
from matchmaker.multiprocess_input_pipeline import *
from matchmaker.performance_monitor import *
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
    secondary_output = {}
    fill_cache=False
    cached_batches = None

    concated_sequences = False
    if config["model"] == "bert_cls":
        concated_sequences = True
        torch.cuda.empty_cache()
    try:
        if use_cache:
            global evaluate_cache
            if eval_tsv not in evaluate_cache:
                fill_cache=True
                evaluate_cache[eval_tsv] = []
            cached_batches = evaluate_cache[eval_tsv]
        
        if not use_cache or fill_cache:
            validation_queue, validation_processes, validation_exit = get_multiprocess_batch_queue("eval-batches",
                                                                                                   multiprocess_validation_loader,
                                                                                                   glob.glob(eval_tsv),
                                                                                                   config,
                                                                                                   _logger,
                                                                                                   queue_size=200)
            #time.sleep(len(validation_processes))  # fill the queue
            _logger.info("[eval_model] --- Start validation with queue.size:" + str(validation_queue.qsize()))
        else:
            _logger.info("[eval_model] --- Start validation with cache size:" + str(len(cached_batches)))

        with torch.no_grad():
            pair_count = 0

            progress = Tqdm.tqdm()
            i=0
            while True: # (not use_cache or fill_cache) or (i<len(cached_batches)): # always true for use queue, for cache: true if there are cached batches :)

                if not use_cache or fill_cache:
                    batch_orig = validation_queue.get()
                    if fill_cache:
                        cached_batches.append(batch_orig)
                else:
                    batch_orig = cached_batches[i]

                if batch_orig == None: # works for both the queue and the cache -> because the last None is also cached :)
                    break

                batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)

                if concated_sequences:
                    torch.cuda.empty_cache()
                    output = model.forward(batch["doc_tokens"],output_secondary_output)
                else:
                    output = model.forward(batch["query_tokens"], batch["doc_tokens"],output_secondary_output)

                if output_secondary_output:
                    output, secondary = output
                    secondary_cpu = {}
                    for tens_name,tens in secondary.items():
                        secondary_cpu[tens_name] = tens.cpu()

                output = output.cpu()  # get the output back to the cpu - in one piece
                #_logger.info("val_output" + str(i))

                for sample_i, sample_query_id in enumerate(batch_orig["query_id"]):  # operate on cpu memory

                    #sample_query_id = int(sample_query_id)
                    sample_doc_id = batch_orig["doc_id"][sample_i]  # again operate on cpu memory

                    if sample_query_id not in validation_results:
                        validation_results[sample_query_id] = []
                        if output_secondary_output:
                            secondary_output[sample_query_id] = {}

                    pair_count +=1
                    validation_results[sample_query_id].append((sample_doc_id, float(output[sample_i])))

                    if output_secondary_output:
                        secondary_i = {}
                        for tens_name,tens in secondary_cpu.items():
                            secondary_i[tens_name] = tens[sample_i].data.numpy()
                        secondary_output[sample_query_id][sample_doc_id] = secondary_i

                #if not use_cache or fill_cache and i % 100 == 0: # only to check for performance regresion
                #    if validation_queue.qsize() < 10:
                #        _logger.warning("validation_queue.qsize() < 10")
                progress.update()
                i+=1
            #_logger.info("val_close")
            progress.close()
        if not use_cache or fill_cache:
            # make sure we didn't make a mistake in the configuration / data preparation
            if validation_queue.qsize() != 0:
                _logger.error("validation_queue.qsize() is not empty after evaluation")

            validation_exit.set()  # allow sub-processes to exit

    except BaseException as e:
        _logger.info('-' * 89)
        _logger.exception('[eval_model] Got exception: ')
        print("----- Attention! - something went wrong in eval_model (see logger) ----- ")
        
        if not use_cache or fill_cache:
            for proc in validation_processes:
                if proc.is_alive():
                    proc.terminate()
        raise e

    return validation_results,pair_count,secondary_output

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
                   output_secondary_output: bool = False):

    if val_name == "cont":
        evaluation_id = str(epoch_number) + "-" +str(batch_number)
    else:
        evaluation_id = "end-" + val_name

    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("cont_val_inference")

    validation_results,pair_count,secondary_output = evaluate_model(model,config,_logger,cuda_device,validation_config["tsv"],use_cache,output_secondary_output)
    
    perf_monitor.stop_block("cont_val_inference",pair_count)

    #_logger.info("val_before_unrolled")
    if val_name != "cont":
        validation_file_path = os.path.join(run_folder, "val-"+evaluation_id+"-output.txt")
        save_sorted_results(validation_results, validation_file_path)

    #perf_monitor.start_block("new_core_metrics")
    ranked_results = unrolled_to_ranked_result(validation_results)
    #new_metrics = calculate_metrics_along_candidate_depth(ranked,load_qrels(validation_config["qrels"]),candidate_set,validation_config["candidate_set_from_to"])
    #new_metrics2 = calculate_metrics_single_candidate_threshold(ranked_results,load_qrels(validation_config["qrels"]),candidate_set,10)
    #new_metrics3 = calculate_metrics_single_candidate_threshold(ranked_results,load_qrels(validation_config["qrels"]),candidate_set,35)
    #new_metrics4 = calculate_metrics_plain(ranked_results,load_qrels(validation_config["qrels"]))
    #perf_monitor.stop_block("new_core_metrics",1)

    #perf_monitor.start_block("old_metrics")

    #
    # write out secondary output
    #
    if output_secondary_output:
        save_secondary_output(model,os.path.join(run_folder, "secondary-"+evaluation_id+".npz"),ranked_results,secondary_output,config["secondary_output"]["top_n"])

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

        metrics = calculate_metrics_along_candidate_depth(ranked_results,load_qrels(validation_config["qrels"]),candidate_set,validation_config["candidate_set_from_to"])
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
        metrics = calculate_metrics_plain(ranked_results,load_qrels(validation_config["qrels"]))
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

    return best_metric_info, metrics, pair_count

#
# test a model after training + save results, metrics 
#
def test_model(model, config,test_config, run_folder, _logger, cuda_device, test_name,candidate_set = None, candidate_set_n = None,output_secondary_output=False):

    test_results,_,secondary_output = evaluate_model(model,config,_logger,cuda_device,test_config["tsv"],output_secondary_output=output_secondary_output)

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
            metrics = calculate_metrics_single_candidate_threshold(ranked_results,load_qrels(test_config["qrels"]),candidate_set,candidate_set_n)

        else:
            metrics = calculate_metrics_plain(ranked_results,load_qrels(test_config["qrels"]))

        metric_file_path = os.path.join(run_folder, test_name+"-metrics.csv")
        save_fullmetrics_oneN(metric_file_path, metrics, -1, -1)

    if output_secondary_output:
        ranked_results = unrolled_to_ranked_result(test_results)
        save_secondary_output(model,os.path.join(run_folder, "secondary-"+test_name+".npz"),ranked_results,secondary_output,config["secondary_output"]["top_n"])

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

    numpy.savez_compressed(out_file,model_data=model_data_secondary_cpu, qd_data=filtered_secondary_output)




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
