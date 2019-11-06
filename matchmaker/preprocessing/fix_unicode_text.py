import ftfy
import sys
from tqdm import tqdm

filename = sys.argv[1]
filename_out = sys.argv[2]
with open(filename, 'r') as f:
    with open(filename_out, 'w') as w:
        for l in tqdm(f):
            # ftfy might introduce additional line breaks & tabs (!) from html sequences
            splits = l.split("\t")

            w.write("\t".join([ftfy.fix_text(split).replace("\n"," ").replace("\t"," ").rstrip() for split in splits])+"\n")