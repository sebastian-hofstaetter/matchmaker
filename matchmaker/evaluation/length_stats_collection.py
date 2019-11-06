#
# generate length stats for a collection.tsv input file
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())


from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1


import matplotlib.pyplot as plt
import numpy as np
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer
import statistics

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store', dest='file',
                    help='tsv collection file', required=True)

parser.add_argument('--results', action='store', dest='result_file',
                    help='output file file', required=True)

args = parser.parse_args()


#
# load data & match & output
# -------------------------------
#  

loader = IrTupleDatasetReader(lazy=True,source_tokenizer=BlingFireTokenizer(),target_tokenizer=BlingFireTokenizer(),lowercase=True)
instances = loader.read(args.file)

doc_lengths = []
doc_lengths_char = []

for i in Tqdm.tqdm(instances):

    doc_lengths.append(len(i["target_tokens"]))
    doc_lengths_char.append(sum([len(token.text) + 1 for token in i["target_tokens"]]) - 1)

result_file = open(args.result_file,"w")

def print_n_show(data_name, lengths):

    result_file.write("# "+data_name+"\n")

    result_file.write("min     " + str(min(lengths))+"\n")
    result_file.write("max     " + str(max(lengths))+"\n")
    result_file.write("mean    " + str(statistics.mean(lengths))+"\n")
    result_file.write("median  " + str(statistics.median(lengths))+"\n")
    result_file.write("stddev  " + str(statistics.stdev(lengths))+"\n")

    result_file.write("50-perc " + str(np.percentile(lengths, 50))+"\n")
    result_file.write("80-perc " + str(np.percentile(lengths, 80))+"\n")
    result_file.write("90-perc " + str(np.percentile(lengths, 90))+"\n")
    result_file.write("95-perc " + str(np.percentile(lengths, 95))+"\n")
    result_file.write("99-perc " + str(np.percentile(lengths, 99))+"\n")

    hist_data,hist_label = np.histogram(lengths, bins=30)
    result_file.write("Histogram: \n")

    for i,data in enumerate(hist_data):
        result_file.write(str(round(hist_label[i],1))+"-"+str(round(hist_label[i + 1],1))+"\t"+ str(int(data))+"\t"+str(round(data/sum(hist_data),3))+"\n")
    result_file.write(" ------- "+"\n"+"\n")


print_n_show("Doc tokens",doc_lengths)
print_n_show("Doc chars",doc_lengths_char)

result_file.close()