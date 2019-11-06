#
# reduce the output file to top 10 per query (guided by cs@n & first stage candidate file)
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import statistics

from matchmaker.utils import *

from matchmaker.evaluation.msmarco_eval import *
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--res-in', action='store', dest='res_in',
                    help='result file in', required=True)

parser.add_argument('--res-out', action='store', dest='res_out',
                    help='result file out', required=True)
                    
parser.add_argument('--candidate-file', action='store', dest='candidate_file',
                    help='trec ranking file location (lucene output)', required=False)

parser.add_argument('--max-rank', action='store', dest='top_n',type=int,
                    help='how many docs per query to retain', required=True)

parser.add_argument('--candidate-rank', action='store', dest='cs_n',type=int,
                    help='how many docs are "re-ranked"', required=False)

parser.add_argument('--run-id', action='store', dest='run_id',
                    help='identifier for each line', required=True)

args = parser.parse_args()


#
# compare (different mrr gains up,same,down)
# -------------------------------
#  
res = load_candidate_from_stream_with_score(open(args.res_in,"r"))#,space_for_rank=30000)

candidate_set = None
if args.candidate_file:
    candidate_set = parse_candidate_set(args.candidate_file,1000)

with open(args.res_out,"w",encoding="utf8") as res_out:
    for query,data in Tqdm.tqdm(res.items()):
        out_count = 0
        for (pid,rank,score) in data:
            if out_count == args.top_n:
                break

            if candidate_set is not None:
                if candidate_set[query][pid] <= args.cs_n:
                    res_out.write(str(query)+" Q0 "+str(pid)+" "+str(out_count)+" "+str(score)+" "+args.run_id+"\n")
                    out_count+=1
            else:
                res_out.write(str(query)+" Q0 "+str(pid)+" "+str(out_count)+" "+str(score)+" "+args.run_id+"\n")
                out_count+=1
