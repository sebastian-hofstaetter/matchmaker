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

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--docs-in', action='store', dest='docs_in', required=True)
parser.add_argument('--docs-out', action='store', dest='docs_out', required=True)

args = parser.parse_args()

max_words_per_block = 130
min_words_per_block = 80
max_blocks_per_doc = 60


stats = defaultdict(int)

with open(args.docs_out,"w",encoding="utf8") as docs_out:

    with open(args.docs_in,"r",encoding="utf8") as in_file:
        for line in tqdm(in_file):
            line = line.split("\t")

            doc_sequence = text_to_sentences(line[2][:200_000]).split("\n") # text

            doc_sequence.insert(0, line[1]) # title

            current_text = ""
            current_word_count = 0              

            if len(doc_sequence) == 1:
                stats["single_sent"]+=1
            #print(doc_sequence)
            i = 0
            for sent_idx, sent in enumerate(doc_sequence): 
                words = sent.split()
                #print(len(words))
                #print(words)

                if current_word_count + len(words) < max_words_per_block:
                    current_text += " ".join(words) + " "
                    current_word_count += len(words)
                    stats["simple_append"]+=1

                elif current_word_count >= min_words_per_block \
                    and len(words) <= max_words_per_block: #current_word_count + len(words) > max_words_per_block \

                    if current_word_count > max_words_per_block:
                        print("ERR1",current_word_count,current_text,sent)

                    docs_out.write(line[0]+"_"+str(i)+"\t"+current_text.strip()+"\n")
                    i+=1
                    #block_lengths.append(current_word_count)
                    #docs[line[0]].append(current_text)
                    current_text = ""
                    current_word_count = 0
                    stats["flush_n_append"]+=1

                    # flush
                    if i == max_blocks_per_doc:
                        break

                    current_text += " ".join(words) + " "
                    current_word_count += len(words)

                else:
                    breaker=False
                    stats["word_split"]+=1

                    while len(words) > 0:

                        set_words = words[:max(0,max_words_per_block - current_word_count)]
                        current_text += " ".join(set_words) + " "
                        current_word_count += len(set_words)
                        words = words[len(set_words):]

                        if current_word_count >= min_words_per_block or \
                           (len(words) == 0 and sent_idx == len(doc_sequence) - 1):
                           #len(words) + current_word_count < min_words_per_block \
                           #or len(words) + current_word_count >= max_words_per_block:

                            # flush
                            if current_word_count > max_words_per_block:
                                print("ERR2",current_word_count,current_text,sent)

                            docs_out.write(line[0]+"_"+str(i)+"\t"+current_text.strip()+"\n")
                            i+=1
                            #block_lengths.append(current_word_count)
                            #docs[line[0]].append(current_text)
                            current_text = ""
                            current_word_count = 0
                            if i == max_blocks_per_doc:
                                breaker=True
                                break

                    if breaker == True:
                        break

for key, val in stats.items():
    print(f"{key}\t{val}")