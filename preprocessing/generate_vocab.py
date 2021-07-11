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
from collections import defaultdict

from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-dir', action='store', dest='out_dir',
                    help='vocab save directory', required=True)

parser.add_argument('--lowercase', action='store', dest='lowercase',type=bool,default=True,
                    help='bool ', required=False)

parser.add_argument('--dataset-files', nargs='+', action='store', dest='dataset_files',
                    help='list of all dataset files: format <id>\t<sequence text>', required=True)

parser.add_argument('--type', action='store', dest='type',
                    help='allenlp or bpe', required=True)

args = parser.parse_args()


#
# load data & create vocab
# -------------------------------
#  

if args.type=="allennlp":
    loader = IrTupleDatasetReader(lazy=True,source_tokenizer=BlingFireTokenizer(),target_tokenizer=BlingFireTokenizer(),lowercase=args.lowercase)

    def getInstances():
        for file in args.dataset_files:
            instances = loader.read(file)
            for i in instances:
                yield Instance({"text":i["target_tokens"]})

    namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for instance in Tqdm.tqdm(getInstances()):
        instance.count_vocab_items(namespace_token_counts)

    for count in [5,10,25,50,100]:
        vocab = Vocabulary(namespace_token_counts, min_count={"tokens":count})
        vocab.save_to_files(args.out_dir + str(count))

    vocab = Vocabulary(namespace_token_counts, min_count={"tokens":1})
    vocab.save_to_files(args.out_dir+"full")

if args.type=="bpe":

    from tokenizers import CharBPETokenizer

    # Initialize a tokenizer
    tokenizer = CharBPETokenizer(lowercase=True)

    # Customize training
    tokenizer.train(files=args.dataset_files, vocab_size=20_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save(args.out_dir,"char-bpe-msmarco-20k")