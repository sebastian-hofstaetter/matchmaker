import argparse
import os
import sys
sys.path.append(os.getcwd())
import glob

from matchmaker.evaluation.msmarco_eval import *

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--results', action='store', dest='results',
                    help='glob to output files', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='csv of metrics per file', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only check judged queries', required=True)

args = parser.parse_args()

#
# work
#

qrels = load_reference(args.qrel)
result_files = glob.glob(args.results)


best_file = ""
best_mrr = 0

with open(args.out_file,"w",encoding="utf8") as metric_file:
    for i,res_file in enumerate(result_files):
        try:
            res = load_candidate(res_file)
            metrics = compute_metrics(qrels, res)

            if i == 0:
                metric_file.write("sep=;\nFile Path; File Name;" + ";".join(k for k, v in metrics.items())+"\n")
            metric_file.write(res_file + ";" +os.path.basename(res_file)+ ";" + ";".join(str(v) for k, v in metrics.items())+"\n")

            if metrics["MRR"] > best_mrr:
                best_file = res_file
                best_mrr = metrics["MRR"] 
                
        except BaseException as e:
            print("got error for:",res_file,e)

print("best_file",best_file)
print("best_mrr",best_mrr)
