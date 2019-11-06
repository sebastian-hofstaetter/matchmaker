import argparse
import os
import sys
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *
from matchmaker.utils import *
from scipy.stats import ttest_rel
import tqdm

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--bm25', action='store', dest='bm25',
                    help='baseline', required=True)

parser.add_argument('--results', action='store', dest='results', nargs='+',
                    help='glob to output files', required=True)

parser.add_argument('--results-cs-n', action='store', dest='cs_at_n', nargs='+',type=int,
                    help='glob to output files', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='csv of metrics per file', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only check judged queries', required=True)

args = parser.parse_args()

#
# eval
#

qrels = load_reference(args.qrel)

candidate_set = parse_candidate_set(args.bm25, max(args.cs_at_n))

rr_per_res = {}
mrr_per_res = {}
recall_per_res = {}
recall_1_per_res = {}

for config_i,neural_res in tqdm.tqdm(enumerate(args.results)):
    res = load_candidate(neural_res)
    dir_name = os.path.basename(os.path.dirname(neural_res))

    qids_to_ranked_candidate_passages = {}
    for query,rank_list in res.items(): #qid_to_ranked_candidate_passages[qid][rank-1]=pid
        qids_to_ranked_candidate_passages[query] = [0] * 1000
        added = 0
        for rank, pid in enumerate(rank_list):
            if pid == 0: # 0 means no more entries > 0 
                break
            if pid in candidate_set[query] and candidate_set[query][pid] <= args.cs_at_n[config_i]:
                qids_to_ranked_candidate_passages[query][added] = pid
                added += 1


    MRR = 0
    RR_perquery = []
    PrcRecall_perquery = []
    qids_with_relevant_passages = 0
    ranking = []
    
    for qid in sorted(qids_to_ranked_candidate_passages):
        if qid in qrels:
            target_pid = qrels[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            ranking.append(0)
            RR_perquery.append(0)
            PrcRecall_perquery.append(0) #len([x for x in candidate_pid[:10] if x in target_pid]), len(target_pid))
            for i in range(0, 10):
                if candidate_pid[i] in target_pid:
                    MRR += 1/(i + 1)
                    RR_perquery.pop()
                    RR_perquery.append( 1/(i + 1))
                    ranking.pop()
                    ranking.append(i+1)
                    PrcRecall_perquery.pop()
                    PrcRecall_perquery.append(1)
                    break

    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR/len(ranking)

    rr_per_res[dir_name] = RR_perquery
    mrr_per_res[dir_name] = MRR

    recall_per_res[dir_name] = PrcRecall_perquery
    recall_1_per_res[dir_name] =  sum((1 for x in ranking if x > 0)) / len(ranking)

    #break
#
# sig test
#
print("got all results")

with open(args.out_file+"-recall.csv","w",encoding="utf8") as rec_metric_file:
    with open(args.out_file+"-mrr.csv","w",encoding="utf8") as mrr_metric_file:
        #headers
        rec_metric_file.write("sep=;\n;")
        mrr_metric_file.write("sep=;\n;")
        for i,res_file in enumerate(rr_per_res):

            rec_metric_file.write(res_file+";;;")
            mrr_metric_file.write(res_file+";;;")
        
        rec_metric_file.write("\n")
        mrr_metric_file.write("\n")

        for i,res_file in enumerate(rr_per_res):
            
            rec_metric_file.write(res_file+";")
            mrr_metric_file.write(res_file+";")

            for t,res_file_other in enumerate(rr_per_res):
                
                if res_file == res_file_other:
                    rec_metric_file.write(";;;")
                    mrr_metric_file.write(";;;")
                    continue
                #
                # mrr
                #
                mrr_res = mrr_per_res[res_file]
                rr_1 = rr_per_res[res_file]
                rr_2 = rr_per_res[res_file_other]

                stat, p_val = ttest_rel(rr_1, rr_2)
                if p_val < 0.05: 
                    s = "s" 
                else: 
                    s =  ""
                #if mrr_res > mrr_per_res[res_file_other]:
                mrr_metric_file.write(str(mrr_res)+";"+s+";"+str(p_val)+";")
                #else:
                #    mrr_metric_file.write(";;;")

                #
                # recall
                #
                rec_res = recall_1_per_res[res_file]
                rec_1 = recall_per_res[res_file]
                rec_2 = recall_per_res[res_file_other]

                stat, p_val = ttest_rel(rec_1, rec_2)
                if p_val < 0.05: 
                    s = "s" 
                else: 
                    s =  ""
                #if rec_res > recall_1_per_res[res_file_other]:
                rec_metric_file.write(str(rec_res)+";"+s+";"+str(p_val)+";")
                #else:
                #    rec_metric_file.write(";;;")

            rec_metric_file.write("\n")
            mrr_metric_file.write("\n")