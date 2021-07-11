from trec_car.read_data import *
import argparse
from tqdm import tqdm

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='trec car paragraph collection file', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='msmarco-passage style collection: id \\t text', required=True)

args = parser.parse_args()


#
# work 
#

with open(args.out_file,"w",encoding="utf8") as out_file:
    for p in tqdm(iter_paragraphs(open(args.in_file,"rb"))):
        out_file.write(p.para_id + "\t" +p.get_text().replace("\t"," ").replace("\n"," ")+"\n")
