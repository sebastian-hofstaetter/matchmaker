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
import torch
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from matchmaker.evaluation.msmarco_eval import *
from matchmaker.utils import *

from matchmaker.models.knrm import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--orig-config', action='store', dest='orig_conf',
                    help='training file location', required=True)

parser.add_argument('--model-file', action='store', dest='model_file',
                    help='training file location', required=True)


args = parser.parse_args()
config = get_config(args.orig_conf)

#
# load data & match & output
# -------------------------------
#  

if config["model"] == "knrm":
    model = KNRM(BasicTextFieldEmbedder({"tokens":None}), None, n_kernels=config["knrm_kernels"], cuda_device=-1)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()


test = 0