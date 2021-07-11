#
# parse trec-format robust 04 into nice tsv documents in a single file
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
                    help='robust 04 files doc collection', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='tsv (id<tab>text) collection format', required=True)

args = parser.parse_args()


#
# work 
#


all_in_files = glob.glob(args.in_file,recursive=True)

text_lengths=[]

filtered_tags=["DOCNO","HT","DATE1","PARENT","PROFILE","DATE","DOCID","LENGTH","AU"]
filtered_line_starts = ("Article Type:",
"DUE TO COPYRIGHT OR OTHER RESTRICTIONS",
"(DUE TO COPYRIGHT OR OTHER RESTRICTIONS",
"ITEM IS INTENDED FOR USE ONLY BY U.S. GOVERNMENT ",
"CONSUMERS. IT IS BASED ON FOREIGN MEDIA CONTENT AND ",
"BEHAVIOR AND IS ISSUED WITHOUT COORDINATION WITH OTHER ",
"U.S. GOVERNMENT COMPONENTS.",
"Document Type:",
"Language:")

replacement_strings=["\n","\t","100>","101>","102>","103>","104>","105>","106>","107>","[Text]"]

with open(args.out_file,"w",encoding="utf8") as out_file:

    for f in tqdm(all_in_files):
        if not os.path.isfile(f):
            continue
        with open(f,"r",encoding="latin-1") as in_file:
            orig = "\n".join([a for a in list(ftfy.fix_file(in_file,normalization="NFKC")) if not a.startswith(filtered_line_starts)])
            soup = BeautifulSoup("<root>\n"+orig+"\n<root>", "lxml-xml")
            for doc in soup.find_all("DOC"):

                id = doc.find("DOCNO").contents[0].strip()
                for t in filtered_tags:
                    for m in doc.find_all(t):
                        m.replace_with("")

                text = doc.get_text()
                for r in replacement_strings:
                    text = text.replace(r," ")

                text = " ".join(text.split())

                out_file.write(id+"\t"+text+"\n")

                text_words = len(text.split())
                text_lengths.append(min(4_000,text_words))





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
