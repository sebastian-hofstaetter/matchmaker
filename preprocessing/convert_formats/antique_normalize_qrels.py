#
# reduce antique qrel scores by 1 
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output file', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='input file', required=True)

args = parser.parse_args()

with open(args.out_file,"w",encoding="utf8") as out_file, \
    open(args.in_file,"r",encoding="utf8") as in_file:

    for line in tqdm(in_file):
        line = line.split()
        #line[3] = str(int(line[3]) - 1)
        line[3] = str(max(int(line[3]) - 2,0))
        out_file.write(" ".join(line)+"\n")