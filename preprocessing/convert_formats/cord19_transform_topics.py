from xml.dom import minidom
import argparse
from tqdm import tqdm

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='trec topic file', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='trec-dl style queries: id \\t text', required=True)

args = parser.parse_args()


#
# work 
#

extract_queries = True
extract_questions = True
extract_narrative = False

with open(args.in_file,'r') as f:
    xmldoc = minidom.parse(f)

with open(args.out_file,"w",encoding="utf8") as out_file:

    for child in xmldoc.documentElement.childNodes: 
        if child.nodeName == 'topic':
            topic_id = child.attributes["number"].nodeValue
            topic_text = ""
            for cc in child.childNodes:
                if cc.nodeName == "query" and extract_queries:
                    topic_text += cc.firstChild.nodeValue + " "
                if cc.nodeName == "question" and extract_questions:
                    topic_text += cc.firstChild.nodeValue + " "
                if cc.nodeName == "narrative" and extract_narrative:
                    topic_text += cc.firstChild.nodeValue + " "

            out_file.write(topic_id + "\t" +topic_text.replace("\t"," ").replace("\n"," ").strip()+"\n")
