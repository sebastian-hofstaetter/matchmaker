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

#from matchmaker.utils import *

#from matchmaker.evaluation.msmarco_eval import *
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
                    
#parser.add_argument('--candidate-file', action='store', dest='candidate_file',
#                    help='trec ranking file location (lucene output)', required=True)

parser.add_argument('--max-rank', action='store', dest='top_n',type=int,
                    help='how many docs per query to retain', required=True)

args = parser.parse_args()


#
# compare (different mrr gains up,same,down)
# -------------------------------
#  
max_out_rank = args.top_n
#max_cs_n = args.top_n


#res = load_candidate(args.res_in,space_for_rank=30000)
#candidate_set = parse_candidate_set(args.candidate_file)


with open(args.res_out,"w",encoding="utf8") as res_out:
    with open(args.res_in,"r",encoding="utf8") as res_in:


        for l in Tqdm.tqdm(res_in):
            l_split = l.strip().split()

            if len(l_split) == 4: # own format
                rank = int(l_split[2])
            elif len(l_split) == 6: # original trec format
                rank = int(l_split[3])

            if rank <= max_out_rank:
                res_out.write(l)

    #for query,data in Tqdm.tqdm(res.items()):
    #    out_count = 0
    #    for (pid,rank,score) in data:
    #        if out_count == max_out_rank:
    #            break
#
    #        #if candidate_set[query][pid] <= max_cs_n:
    #        res_out.write(str(query)+" "+str(pid)+" "+str(rank)+" "+str(score)+"\n")
    #        out_count+=1
#
