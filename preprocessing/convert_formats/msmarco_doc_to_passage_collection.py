#
# msmarco doc collection format to passage (fewer columns)
# 
# * doc collection.tsv: docid, url, title, body
# * passage collection.tsv: docid, concat(title,body)
#  
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import Tqdm
Tqdm.default_mininterval = 1
import numpy

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='doc collection', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='passage collection format', required=True)

args = parser.parse_args()


#
# work 
#

title_lengths=[]
text_lengths=[]

with open(args.in_file,"r",encoding="utf8") as in_file:
    with open(args.out_file,"w",encoding="utf8") as out_file:
        for line in tqdm(in_file):

            ls = line.split("\t") # docid, url, title, body
            out_file.write(ls[0]+"\t"+ls[2]+"\t"+ls[3]) # \n is part of ls[3]

            text_words = len(ls[3].split())
            text_lengths.append(min(4_000,text_words))
            title_lengths.append(len(ls[2].split()))

def crappyhist(a, bins=20, width=30,range_=(0,1)):
    h, b = numpy.histogram(a, bins,range_)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i], 
            '#'*int(width*h[i]/numpy.amax(h)), 
            h[i],#/len(a), 
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

print("Text")
crappyhist(text_lengths,range_=(0,4_000))
print("Title")
crappyhist(title_lengths,range_=(0,40))