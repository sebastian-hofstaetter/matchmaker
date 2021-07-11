import argparse
from tqdm import tqdm
import csv
import json
import os
from collections import defaultdict

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--metadata', action='store', dest='metadata',
                    help='cord-19 metadata file', required=True)

parser.add_argument('--ft-directory', action='store', dest='ft_directory',
                    help='cord-19 fulltext directory', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='trec-dl style collection: id \\t text', required=True)

args = parser.parse_args()


#
# work 
#
stats = defaultdict(int)

def parse_fulltext(file):
    if not os.path.exists(file):
        stats["not_found"]+=1
        return None

    with open(file,"r",encoding="utf8") as ft_file:
        data = json.load(ft_file)

        text = ""
        for block in data["body_text"]:
            text += block["section"] + " " + block["text"] + " "
        for block in data["ref_entries"].values():
            text += block["text"] + " "
    return text


with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.metadata,"r",encoding="utf8") as metadata:

        reader = csv.DictReader(metadata)
        for meta_line in tqdm(reader):
            stats["metadata_line_count"]+=1
            # doc_id, title, abstract from metadata
            doc_text = meta_line["title"]+" "+meta_line["abstract"]

            # get fulltext from json files (if avail)

            #get_ft = meta_line['has_pmc_xml_parse'] == "True" or meta_line['has_pdf_parse'] == "True"
#
            #if get_ft:
            #    ft = parse_fulltext(os.path.join(args.ft_directory,"pmc_json",meta_line["pmcid"]+".xml.json"))
            #    if ft is not None:
            #        stats["pmc_fulltext"]+=1
            #        doc_text += ft
            #        get_ft == False
            #        
            #if get_ft:
            #    ft = parse_fulltext(os.path.join(args.ft_directory,"pdf_json",meta_line["sha"]+".json"))
            #    if ft is not None:
            #        stats["pdf_parse"]+=1
            #        doc_text += ft

            #doc_text = " ".join([t for t in doc_text.split() if len(t) < 60])
            if len(doc_text.strip())>0:
                out_file.write(meta_line["cord_uid"] + "\t" +doc_text.replace("\t"," ").replace("\n"," ").strip()[:100_000]+"\n")
            else:
                stats["empty_file"]+=1

for key, val in stats.items():
    print(f"{key}\t{val}")