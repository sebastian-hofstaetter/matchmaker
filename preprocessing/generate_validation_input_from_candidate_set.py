#
# generate validation.tsv tuples from candidate set files 
# (for example lucene trec output for bm25 et al.)
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())


from allennlp.common import Tqdm
Tqdm.default_mininterval = 1
from blingfire import *

import random
random.seed(208973249)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='val_out_file',
                    help='validation output file location', required=True)

parser.add_argument('--candidate-file', action='store', dest='candidate_file',
                    help='trec ranking file location (lucene output)', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

parser.add_argument('--top-N', action='store', dest='top_n',type=int,
                    help='how many docs per query to take', required=True)

parser.add_argument("--add-scores", help="add bm25 scores to output",default=False,
                    action="store_true")

args = parser.parse_args()

max_doc_char_length = 100_000


#
# load data 
# -------------------------------
# 
collection = {} # int id -> full line dictionary
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        collection[_id] = ls[1].rstrip()[:max_doc_char_length] #+"\t"+ls[2].rstrip()[:max_doc_char_length]

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = ls[0]
        queries[_id] = ls[1].rstrip()


def data_augment(aug_type,string):

    if aug_type == "shuffle_sent":
        doc_sequence = text_to_sentences(string).split("\n")
        # fall back to rotate in case of only one sentence
        #if len(doc_sequence) < 3:
        #    tokens = text_to_words(string).split()
        #    n = random.randint(0,len(tokens)-1)
        #    doc_sequence = " ".join(tokens[n:] + tokens[:n])
        #else:
        random.shuffle(doc_sequence)
        doc_sequence = " ".join(doc_sequence)

    elif aug_type == "reverse_sent":
        doc_sequence = text_to_sentences(string).split("\n")
        doc_sequence = " ".join(doc_sequence[::-1])

    elif aug_type == "rotate":
        tokens = text_to_words(string).split()
        n = random.randint(0,len(tokens)-1)
        doc_sequence = " ".join(tokens[n:] + tokens[:n])

    elif aug_type == "none":
        doc_sequence = string
    else:
        raise Exception("wrong aug_type")

    return doc_sequence

#
# produce output
# -------------------------------
#  
max_rank = args.top_n

with open(args.val_out_file,"w",encoding="utf8") as val_out_file:
    with open(args.candidate_file,"r",encoding="utf8") as candidate_file:

        for line in tqdm(candidate_file):
            ls = line.split() # 2 Q0 1782337 1 21.656799 Anserini

            if len(ls) == 4:
                query_id = ls[0]
                doc_id = ls[1]
                rank = ls[2]
            else:
                query_id = ls[0]
                doc_id = ls[2]
                rank = ls[3]


            if int(rank) > max_rank:
                continue

            
            if query_id in queries:
                
                #augment data 
                #doc_sequence = data_augment("shuffle_sent",collection[doc_id])
                #doc_sequence = data_augment("reverse_sent",collection[doc_id])
                #doc_sequence = data_augment("rotate",collection[doc_id])
                doc_sequence = collection[doc_id]

                if args.add_scores:
                    out_arr = [query_id,doc_id,queries[query_id],doc_sequence,ls[4]]
                else:
                    out_arr = [query_id,doc_id,queries[query_id],doc_sequence]

                val_out_file.write("\t".join(out_arr)+"\n")