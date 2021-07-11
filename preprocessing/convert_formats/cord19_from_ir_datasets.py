import argparse
from tqdm import tqdm
import os

#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output folder', required=False,default="C:\\Users\\sebas\\data\\trec-covid" )

args = parser.parse_args()


#
# work 
#


import ir_datasets
dataset = ir_datasets.load('cord19/trec-covid')

with open(os.path.join(args.out_file,"complete-queries-title.tsv"),"w",encoding="utf8") as out_file,\
     open(os.path.join(args.out_file,"complete-queries-description.tsv"),"w",encoding="utf8") as out_file2:
 for query in dataset.queries_iter():
    #query # namedtuple<query_id, title, description, narrative>
    out_file.write(query.query_id + "\t" +query.title.replace("\t"," ").replace("\n"," ").strip()+"\n")
    out_file2.write(query.query_id + "\t" +query.description.replace("\t"," ").replace("\n"," ").strip()+"\n")

with open(os.path.join(args.out_file,"collection.tsv"),"w",encoding="utf8") as out_file:
 for doc in dataset.docs_iter():
    #doc # namedtuple<doc_id, title, doi, date, abstract>
    out_file.write(doc.doc_id + "\t" +doc.title.replace("\t"," ").replace("\n"," ").strip()+" "+doc.abstract.replace("\t"," ").replace("\n"," ").strip()+"\n")

with open(os.path.join(args.out_file,"complete-qrels.txt"),"w",encoding="utf8") as out_file:
 for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>
    out_file.write(qrel.query_id + " Q0 " +qrel.doc_id+" "+str(qrel.relevance)+"\n")