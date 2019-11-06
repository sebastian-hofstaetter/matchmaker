#
# generate length stats for a ranked output file 
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from matchmaker.dataloaders.ir_labeled_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1


import matplotlib.pyplot as plt
import numpy as np

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--rank-file', action='store', dest='rank_file',
                    help='the models output file with ranks and scores', required=True)

parser.add_argument('--val-file', action='store', dest='val_file',
                    help='validation.tsv location used to produce the rank-file', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel location', required=True)

args = parser.parse_args()


#
# load data & match & output
# -------------------------------
#  

qrel = load_reference(args.qrel)

loader = IrLabeledTupleDatasetReader(lazy=True,tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()))
instances = loader.read(args.val_file)

known_queries = {}
known_docs = {}

for i in Tqdm.tqdm(instances):

    q_id = i["query_id"].label
    d_id = i["doc_id"].label

    if q_id not in known_queries:
        known_queries[q_id] = len(i["query_tokens"])

    if d_id not in known_docs:
        known_docs[d_id] = len(i["doc_tokens"])

ranks = load_candidate(args.rank_file)

doc_lengths = []
doc_lengths_relevant = []

rank_doc_lengths = {}
rank_doc_lengths_relevant = {}

for q_id in ranks:
    for rank, d_id in enumerate(ranks[q_id]):
        if rank > 49:
            continue
        if d_id == 0: 
            continue
        if d_id not in known_docs: 
            print(d_id,"not known") 
            continue

        doc_lengths.append(known_docs[d_id])

        if rank not in rank_doc_lengths:
            rank_doc_lengths[rank] = []
        rank_doc_lengths[rank].append(known_docs[d_id])

        if q_id in qrel:
            if d_id in qrel[q_id]:
                doc_lengths_relevant.append(known_docs[d_id])

                if rank not in rank_doc_lengths_relevant:
                    rank_doc_lengths_relevant[rank] = []
                rank_doc_lengths_relevant[rank].append(known_docs[d_id])

print("rank_doc_lengths_relevant")
for rank, data in sorted(rank_doc_lengths_relevant.items()):
    print(rank,"\t",round(statistics.mean(data),1),"\t",round(statistics.median(data),1),"("+str(len(data))+" docs)")

print("rank_doc_lengths")
for rank, data in rank_doc_lengths.items():
    print(rank,"\t",round(statistics.mean(data),1),"\t",round(statistics.median(data),1))

def print_n_show(data_name, lengths):
    hist_data,hist_label,_ = plt.hist(lengths, bins=30)
    plt.ylabel('# '+data_name)
    plt.xlabel(data_name + ' lengths')
    plt.show()

    print(data_name +" data\n")
    print("Mean",statistics.mean(lengths))
    print("Median",statistics.median(lengths))

    for i,data in enumerate(hist_data):
        print(str(round(hist_label[i],1))+"-"+str(round(hist_label[i + 1],1)),"\t", int(data),"\t",round(data/sum(hist_data),3))
    print(" ------- ")

print_n_show("Doc raw",doc_lengths)
#print_n_show("Query raw",query_lengths)
print_n_show("Doc relevant",doc_lengths_relevant)


input("Press Enter to continue...")
