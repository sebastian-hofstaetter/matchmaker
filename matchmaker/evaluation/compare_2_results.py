#
# compare 2 results from any runs (can be lucene or neural-model)
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt


from matchmaker.evaluation.msmarco_eval import *

from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from msmarco_eval import *

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--res-1', action='store', dest='res_1',
                    help='first result', required=True)

parser.add_argument('--res-2', action='store', dest='res_2',
                    help='second result', required=True)

parser.add_argument('--queries', action='store', dest='query',
                    help='original query text', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only check judged queries', required=True)

parser.add_argument('--out', action='store', dest='out_file',
                    help='out file saving up/down score per query id', required=True)

args = parser.parse_args()


#
# compare (different mrr gains up,same,down)
# -------------------------------
#  

qrels = load_reference(args.qrel)

res_1 = load_candidate(args.res_1)
res_2 = load_candidate(args.res_2)

queries = {}
with open(args.query,"r",encoding="utf8") as query_file:
    for line in query_file:
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        queries[_id] = ls[1].rstrip()#.split()


up_down_per_query = {}
res_1_stat_per_query = {}
res_2_stat_per_query = {}

metric_1 = compute_metrics(qrels,res_1)
metric_2 = compute_metrics(qrels,res_2)

both_not_found_count = 0
only_in_1 = 0
only_in_2 = 0

def get_first_rel(ref,candidate):
    for i, d in enumerate(candidate):
        #if i > 10: break ## mrr @ 10
        if d in ref:
            return i + 1
    return 0

out_file = open(args.out_file,"w")

for query in qrels:
    if query in res_1 and query in res_2:
        res_1_rel = get_first_rel(qrels[query], res_1[query])
        res_2_rel = get_first_rel(qrels[query], res_2[query])

        res_1_stat_per_query[query] = res_1_rel
        res_2_stat_per_query[query] = res_2_rel

        if res_1_rel != 0 and res_2_rel != 0:
            up_down_per_query[query] = res_1_rel - res_2_rel
            out_file.write(str(query)+"\t"+str(res_1_rel)+"\t"+str(res_2_rel)+"\t"+queries[query]+"\n")
        elif res_1_rel != 0:
            only_in_1 += 1
        elif res_2_rel != 0:
            only_in_2 += 1
        else:
            both_not_found_count +=1

out_file.close()

up_queries = {}
down_queries = {}

for q,v in up_down_per_query.items():
    if v > 0: #up
        up_queries[q] = queries[q]
    if v < 0: #down
        down_queries[q] = queries[q]

total = len(up_down_per_query)
up = sum(1 for k,v in up_down_per_query.items() if v > 0)
down = sum(1 for k,v in up_down_per_query.items() if v < 0)
same = sum(1 for k,v in up_down_per_query.items() if v == 0)

avg_up = statistics.median(v for k,v in up_down_per_query.items() if v > 0)
avg_down = statistics.median(v for k,v in up_down_per_query.items() if v < 0)

print("1 metrics:",metric_1)
print("2 metrics:",metric_2)
print("total (only judged & found)",total)
print("non relevant in both",both_not_found_count)
print("only_in_1",only_in_1)
print("only_in_2",only_in_2)
print("---------------")
print(" # up",up,"(",up/total*100,"%)")#"avg q length:",statistics.mean(len(v) for k,v in up_queries.items()))
#for i,(k,v) in enumerate(up_queries.items()):
#    if i==50:break
#    print(" ".join(v))
print(" # down",down,"(",down/total*100,"%)")#,"avg q length:",statistics.mean(len(v) for k,v in down_queries.items()))
#for i,(k,v) in enumerate(down_queries.items()):
#    if i==50:break
#    print(" ".join(v))
print(" # same",same,"(",same/total*100,"%)")
print("---------------")

print(" avg up",avg_up)
print(" avg down",avg_down)

#same_data=[res_2_stat_per_query[k] for k,v in up_down_per_query.items() if v == 0]
#plt.hist(same_data,bins=range(min(same_data), max(same_data) + 1, 1))

#
#data_2=[res_2_stat_per_query[k] for k,v in up_down_per_query.items() if v > 0]
#data_1=[res_2_stat_per_query[k] for k,v in up_down_per_query.items() if v < 0]
#plt.hist([data_1,data_2],bins=range(1, 100, 1),label=["1","2"])
#plt.legend()
#plt.show()