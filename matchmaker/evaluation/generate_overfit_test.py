#
# generate a vocab file for re-use
# -------------------------------
#

# usage:
# python matchmaker/preprocessing/generate_vocab.py --out-dir vocabDir --dataset-files list of all dataset files: format <id>\t<sequence text>'

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

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-out-file', action='store', dest='train_out_file',
                    help='training file location', required=True)

parser.add_argument('--reduced-val-file', action='store', dest='val_out_file',
                    help='training file location', required=True)


parser.add_argument('--qrels', action='store', dest='qrels',
                    help='qrel location', required=True)

parser.add_argument('--validation-file', action='store', dest='validation_file',
                    help='validation.tsv location', required=True)


args = parser.parse_args()


#
# load data & match & output
# -------------------------------
#  

qrel = load_reference(args.qrels)

loader = IrLabeledTupleDatasetReader(lazy=True,tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()))
instances = loader.read(args.validation_file)

per_query_pos_instances = {}
per_query_neg_instances = {}

for i in instances:

    current_id = i["query_id"].label
    is_pos = current_id in qrel and i["doc_id"].label in qrel[current_id]

    if is_pos:
        if current_id not in per_query_pos_instances:
            per_query_pos_instances[current_id]=[]
        per_query_pos_instances[current_id].append(i)
    else:
        if current_id not in per_query_neg_instances:
            per_query_neg_instances[current_id] = []
        per_query_neg_instances[current_id].append(i)

with open(args.train_out_file,"w",encoding="utf8") as train_out_file:
    with open(args.val_out_file,"w",encoding="utf8") as val_out_file:

        for pos_i in per_query_pos_instances:
            if pos_i in per_query_neg_instances:

                train_out_file.write("\t".join([
                    " ".join(t.text for t in per_query_pos_instances[pos_i][0]["query_tokens"].tokens),
                    " ".join(t.text for t in per_query_pos_instances[pos_i][0]["doc_tokens"].tokens),
                    " ".join(t.text for t in per_query_neg_instances[pos_i][0]["doc_tokens"].tokens)])+"\n")

                val_out_file.write("\t".join([
                    str(per_query_pos_instances[pos_i][0]["query_id"].label),
                    str(per_query_pos_instances[pos_i][0]["doc_id"].label),
                    " ".join(t.text for t in per_query_pos_instances[pos_i][0]["query_tokens"]),
                    " ".join(t.text for t in per_query_pos_instances[pos_i][0]["doc_tokens"])])+"\n")

                for neg in per_query_neg_instances[pos_i][:100]:
                    val_out_file.write("\t".join([
                        str(neg["query_id"].label),
                        str(neg["doc_id"].label),
                        " ".join(t.text for t in neg["query_tokens"]),
                        " ".join(t.text for t in neg["doc_tokens"])])+"\n")