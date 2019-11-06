#
# generate length stats for a validation.tsv input file
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

parser.add_argument('--val-file', action='store', dest='val_file',
                    help='validation file location', required=True)

#parser.add_argument('--out-folder', action='store', dest='out_folder',
#                    help='output location', required=True)

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

query_lengths = []
doc_lengths = []

query_lengths_unique = []
doc_lengths_unique = []

doc_lengths_relevant = []

for i in Tqdm.tqdm(instances):

    q_id = i["query_id"].label
    d_id = i["doc_id"].label

    query_lengths.append(len(i["query_tokens"]))
    doc_lengths.append(len(i["doc_tokens"]))

    if q_id not in known_queries:
        known_queries[q_id] = None
        query_lengths_unique.append(len(i["query_tokens"]))

    if d_id not in known_docs:
        known_docs[d_id] = None
        doc_lengths_unique.append(len(i["doc_tokens"]))

    if q_id in qrel:
        if d_id in qrel[q_id]:
            doc_lengths_relevant.append(len(i["doc_tokens"]))


def print_n_show(data_name, lengths):
    hist_data,hist_label,_ = plt.hist(lengths, bins=30)
    plt.ylabel('# '+data_name)
    plt.xlabel(data_name + ' lengths')
    plt.show()

    print(data_name +" data\n")
    for i,data in enumerate(hist_data):
        print(str(round(hist_label[i],1))+"-"+str(round(hist_label[i + 1],1)),"\t", int(data),"\t",round(data/sum(hist_data),3))
    print(" ------- ")

print_n_show("Doc raw",doc_lengths)
#print_n_show("Query raw",query_lengths)

print_n_show("Doc unique",doc_lengths_unique)
print_n_show("Query unique",query_lengths_unique)

print_n_show("Doc relevant",doc_lengths_relevant)


input("Press Enter to continue...")
