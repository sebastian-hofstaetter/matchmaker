import argparse
import os
import sys
sys.path.append(os.getcwd())
from allennlp.common import Tqdm
import glob
from matchmaker.evaluation.msmarco_eval import *

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--result', action='store', dest='results',
                    help='result file', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='csv of metrics per file', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only check judged queries', required=True)

parser.add_argument('--cutoff', action='store', dest='cutoff',type=int,
                    help='@rank cutoff', required=True)


args = parser.parse_args()

#
# work
#

qrels = load_reference(args.qrel)
result_files = glob.glob(args.results)

best_mrr = (0,"",0)
best_relevant = (0,"",0)

with open(args.out_file,"w",encoding="utf8") as metric_file:

    for t,res_file in Tqdm.tqdm(enumerate(result_files)):
        try:
            res_candidate = load_candidate(res_file,args.cutoff)
            for i in Tqdm.tqdm(range(1, args.cutoff)):

                metrics = compute_metrics(qrels,res_candidate , i)
                
                if i == 1 and t == 0:
                    metric_file.write("sep=,\nFile,Cutoff," + ",".join(k for k, v in metrics.items())+"\n")
                    
                if metrics["QueriesWithRelevant"] > best_relevant[0]:
                    best_relevant = (metrics["QueriesWithRelevant"],res_file,i)
                    print("got new best QueriesWithRelevant",best_relevant)
                if metrics["MRR"] > best_mrr[0]:
                    best_mrr = (metrics["MRR"],res_file,i)
                    print("got new best MRR",best_mrr)

                metric_file.write(res_file+","+str(i) + "," + ",".join(str(v) for k, v in metrics.items())+"\n")

        except BaseException as e:
            print("got error for:",res_file,e)