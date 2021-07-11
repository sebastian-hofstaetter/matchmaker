#
# msmarco doc: split the offical train set in a train and validation set   
# -------------------------------
#

import random
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='the official train file', required=True)

parser.add_argument('--out-file-validation', action='store', dest='out_file_validation',
                    help='validation file', required=True)

parser.add_argument('--out-file-train', action='store', dest='out_file_train',
                    help='the new train file', required=True)

args = parser.parse_args()

total_queries = 0
with open(args.in_file,"r",encoding="utf8") as in_file:
    for line in tqdm(enumerate(in_file)):
        total_queries+=1

#
# work 
#
random.seed(42)
validation_id_set = set(random.sample(range(0, total_queries), 5000))

with open(args.in_file,"r",encoding="utf8") as in_file:
    with open(args.out_file_validation,"w",encoding="utf8") as out_file_validation:
        with open(args.out_file_train,"w",encoding="utf8") as out_file_train:
            for i,line in tqdm(enumerate(in_file)):

                if i in validation_id_set:
                    out_file_validation.write(line)
                else:
                    out_file_train.write(line)