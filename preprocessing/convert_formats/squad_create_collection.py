import argparse
from tqdm import tqdm
import json

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='squad file', required=True)

parser.add_argument('--in-file-dev', action='store', dest='in_file_dev',
                    help='squad file', required=True)

parser.add_argument('--out-file-collection', action='store', dest='out_file_coll',
                    help='msmarco-passage style collection: id \\t text', required=True)

parser.add_argument('--out-file-queries', action='store', dest='out_file_queries',
                    help='msmarco-passage style', required=True)

parser.add_argument('--out-file-qrels', action='store', dest='out_file_qrels',
                    help='msmarco-passage style', required=True)

parser.add_argument('--out-file-qrels-qa', action='store', dest='out_file_qrels_qa',
                    help='msmarco-passage style', required=True)

parser.add_argument('--out-file-queries-dev', action='store', dest='out_file_queries_dev',
                    help='msmarco-passage style', required=True)

parser.add_argument('--out-file-qrels-dev', action='store', dest='out_file_qrels_dev',
                    help='msmarco-passage style', required=True)

parser.add_argument('--out-file-qrels-dev-qa', action='store', dest='out_file_qrels_dev_qa',
                    help='msmarco-passage style', required=True)

args = parser.parse_args()


#
# work 
#

data = json.load(open(args.in_file,"r"))

passage_id = 0
query_id = 0

with open(args.out_file_coll,"w",encoding="utf8") as out_file_coll,\
     open(args.out_file_queries,"w",encoding="utf8") as out_file_queries,\
     open(args.out_file_qrels,"w",encoding="utf8") as out_file_qrels,\
     open(args.out_file_qrels_qa,"w",encoding="utf8") as out_file_qrels_qa:

    for paragraph in tqdm([y for x in data["data"] for y in x["paragraphs"]]):

        passage_text = paragraph["context"]

        out_file_coll.write("p-"+str(passage_id) + "\t" +passage_text.replace("\t"," ").replace("\n"," ")+"\n")

        for par_qa in paragraph["qas"]:
            if par_qa["is_impossible"] == False:
                out_file_queries.write("q-"+str(query_id) + "\t" +par_qa["question"].replace("\t"," ").replace("\n"," ")+"\n")
                out_file_qrels.write("q-"+str(query_id) + " Q0 " +"p-"+str(passage_id)+" 1\n")
                answers = []
                answer_texts = []
                for a in par_qa["answers"]:
                    answers.append(str(a["answer_start"])+","+str(a["answer_start"]+len(a["text"])))
                    answer_texts.append(a["text"].replace("\t"," ").replace("\n"," "))
                out_file_qrels_qa.write("q-"+str(query_id) + "\t" +"p-"+str(passage_id)+"\t"+" ".join(answers)+"\t"+"\t".join(answer_texts)+"\n")
                query_id += 1 
        
        passage_id += 1

data = json.load(open(args.in_file_dev,"r"))

with open(args.out_file_coll,"a",encoding="utf8") as out_file_coll,\
     open(args.out_file_queries_dev,"w",encoding="utf8") as out_file_queries,\
     open(args.out_file_qrels_dev,"w",encoding="utf8") as out_file_qrels,\
     open(args.out_file_qrels_dev_qa,"w",encoding="utf8") as out_file_qrels_qa:

    for paragraph in tqdm([y for x in data["data"] for y in x["paragraphs"]]):

        passage_text = paragraph["context"]

        out_file_coll.write("p-"+str(passage_id) + "\t" +passage_text.replace("\t"," ").replace("\n"," ")+"\n")

        for par_qa in paragraph["qas"]:
            if par_qa["is_impossible"] == False:
                out_file_queries.write("q-"+str(query_id) + "\t" +par_qa["question"].replace("\t"," ").replace("\n"," ")+"\n")
                out_file_qrels.write("q-"+str(query_id) + " Q0 " +"p-"+str(passage_id)+" 1\n")
                query_id += 1 
                answers = []
                answer_texts = []
                for a in par_qa["answers"]:
                    loc = str(a["answer_start"])+","+str(a["answer_start"]+len(a["text"]))
                    if loc not in answers:
                        answers.append(loc)
                        answer_texts.append(a["text"].replace("\t"," ").replace("\n"," "))
                out_file_qrels_qa.write("q-"+str(query_id) + "\t" +"p-"+str(passage_id)+"\t"+" ".join(answers)+"\t"+"\t".join(answer_texts)+"\n")

        passage_id += 1

