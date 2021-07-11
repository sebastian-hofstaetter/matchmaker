#
# 
# -------------------------------
#

import random
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())
from blingfire import *
from collections import defaultdict
import numpy
import glob
from langdetect import detect

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--files', action='store', dest='in_files', required=True)
parser.add_argument('--out', action='store', dest='out', required=True)

args = parser.parse_args()

stats = defaultdict(int)
title_lengths=[]
text_lengths=[]
langs=[]

all_files = glob.glob(args.in_files)

with open(args.out,"w",encoding="utf8") as out:

    for f in tqdm(all_files):

        with open(f,"r",encoding="utf8") as in_file:
            try:
                line = in_file.read().split("\t")

                title = line[0]
                text = line[1]
                text_words = len(text.split())
                text_lengths.append(min(4_000,text_words))
                title_lengths.append(len(title.split()))

                if title.strip() != "" and text_words > 130:
                    lang = detect(text)
                    langs.append(lang)

                    if lang == "en":
                        stats["good"]+=1

                        out.write(os.path.basename(f)[:-4]+"\t"+ title+"\t"+text+"\n")
                    else:
                        stats["wrong-lang"]+=1
                else:
                    stats["too-short"]+=1
                    
            except BaseException as e:
                print("Error",e)
            

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

from collections import Counter
print(Counter(langs))

for key, val in stats.items():
    print(f"{key}\t{val}")