#
# parse trec-format tripclick into nice tsv documents in a single file
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

replacement_strings=["\n","\t"] 

import re
import os, sys

def read_trecdocfile(path) :

        f=open(path, 'r', encoding='utf-8')
        docs=[]
        lines=iter(f.read().splitlines())
        for line in lines:
            line = line.strip()

            if len(line) == 0:
                continue

            if (line == '<DOC>'):
                doc = {}
                isinheader = False
                isintext = False
            else:
                try:
                    if '<DOCNO>' in line:
                        m = re.match("<(.*)>(.*)</\\1>", line)
                        if m:
                            if m.group(1)=='DOCNO':
                                doc['docno']=m.group(2).strip()
                        else:
                            line = next(lines)
                            doc['docno'] = line.replace('</DOCNO>', '').strip()
                    #self.logger.info("Reading %s starting with %s"%(path, str(doc['docno'])))
                    elif '<TITLE>' in line:
                        m = re.match("<(.*)>(.*)</\\1>", line)
                        if m:
                            if m.group(1)=='TITLE':
                                doc['title']=m.group(2).strip()
                        else:
                            line = next(lines)
                            doc['title'] = line.replace('</TITLE>', '').strip()
                    elif '<TEXT>' in line:
                        text = line.replace('<TEXT>', '')
                        if '</TEXT>' in line:
                            text=line.replace('</TEXT>', '')
                        else:
                            while (1):
                                line=next(lines)
                                if '</TEXT>' in line:
                                    text += " " + line.replace('</TEXT>', '')
                                    break
                                elif '</DOC>' in line:
                                    isintext=False
                                    break
                                else:
                                    text += " " + line

                        doc['text'] = text.strip()

                        if not isintext:
                            docs.append(doc)
                            continue

                        while (1):
                            line=next(lines)
                            if '</DOC>' in line:
                                docs.append(doc)
                                break
                except(StopIteration):
                    #break and return whatever docs you gathered so far
                    sys.stdout.write("trouble in reading file: %s" % path)
                    sys.stdout.flush()
                    break

        f.close()

        return docs


with open(args.out_file,"w",encoding="utf8") as out_file:

    for f in tqdm(all_in_files):
        if not os.path.isfile(f):
            continue
        docs = read_trecdocfile(f)
        print(f,len(docs))
        for doc in docs:

            id = doc["docno"] 
            text = doc['title'] + " " + doc['text']
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
