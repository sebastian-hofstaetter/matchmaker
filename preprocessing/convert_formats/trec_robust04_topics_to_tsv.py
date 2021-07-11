#
# parse trec-topics into tsv 
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
import glob

from bs4 import BeautifulSoup
import ftfy
#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='robust 04 topic file', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='tsv (id<tab>text) query format', required=True)

parser.add_argument('--type', action='store', dest='type',
                    help='topic or description', required=True)

args = parser.parse_args()


#
# work 
#


all_in_files = glob.glob(args.in_file,recursive=True)

text_lengths=[]


with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.in_file,"r",encoding="latin-1") as in_file:

        current_id = ""
        current_title = ""
        current_desc = ""
        reading_desc=False
        reading_title=False
        for line in in_file:
            if line.startswith("<num> Number: "):
                current_id = line.replace("<num> Number: ","").strip()
            if line.startswith("<title>"):
                reading_title=True
            if line.startswith("<desc>"):
                reading_desc=True
                reading_title=False
            elif line.startswith("<narr>"):
                reading_desc=False
            elif reading_desc:
                current_desc += " " + line

            

            
            if reading_title:
                current_title += line.replace("<title>","")
            
            if line.startswith("</top"):
                
                text = current_title if args.type=="title" else current_desc
                text = " ".join(text.split())
                out_file.write(current_id+"\t"+text.strip()+"\n")

                text_words = len(text.split())
                text_lengths.append(min(3_0,text_words))

                current_id = ""
                current_title = ""
                current_desc = ""

def crappyhist(a, bins=30, width=30,range_=(0,1)):
    h, b = numpy.histogram(a, bins,range_)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i], 
            '#'*int(width*h[i]/numpy.amax(h)), 
            h[i],#/len(a), 
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

print("Text")
crappyhist(text_lengths,range_=(0,3_0))
