import argparse
from collections import defaultdict
import json
import os
import sys
sys.path.append(os.getcwd())

import glob
import csv
from rich.table import Table
from rich.console import Console

from rich import print
from rich.console import RenderGroup
from rich.panel import Panel



#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--dir', action='store', dest='result_dir', #nargs='+',
                    help='glob to csv result files', required=True)
                    
args = parser.parse_args()


print_metrics = ["nDCG@10","MRR@10","Recall@1000"]
print_efficiency_metrics = ["encode","indexing","search_query_encode",
                            "search_nn_lookup","faiss_index_size_on_disk"]

#
# effectiveness metrics
#
console = Console()
if "*" in args.result_dir:
    files = glob.glob(os.path.join(args.result_dir,"*metrics.csv"))
else:
    files = glob.glob(os.path.join(args.result_dir,"**","*metrics.csv"))

basenames = set()
for f in files:
    basenames.add(os.path.basename(f))

for base in basenames:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Start", style="dim")
    table.add_column("Experiment", style="black")
    for m in print_metrics:
        table.add_column(m,justify="right")

    console.print(base, style="bold magenta")
    for f in sorted(files):
        if os.path.basename(f) != base:
            continue
        exp = os.path.basename(os.path.dirname(f))
        with open(f,"r",encoding="utf8") as in_f:
            lines = in_f.readlines()[1:]
            reader = csv.DictReader(lines)
            for row in reader:
                #2021-04-02_0326_
                cols = [exp[:13].replace("_"," ")+":"+exp[13:15],exp[16:]]

                for m in print_metrics:
                    cols += [str(round(float(row[m]),3))]

                #print(cols)
                table.add_row(*cols)

                break


    console.print(table)

#
# efficiency metrics
#
if "*" in args.result_dir:
    files = glob.glob(os.path.join(args.result_dir,"efficiency_metrics.json"))
else:
    files = glob.glob(os.path.join(args.result_dir,"**","efficiency_metrics.json"))

basenames = set()
for f in files:
    basenames.add(os.path.basename(f))

for base in basenames:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Start", style="dim")
    table.add_column("Experiment", style="black")
    for m in print_efficiency_metrics:
        table.add_column(m,justify="right")

    console.print(base, style="bold magenta")
    for f in sorted(files):
        if os.path.basename(f) != base:
            continue
        exp = os.path.basename(os.path.dirname(f))
        with open(f,"r",encoding="utf8") as in_f:
            reader = json.load(in_f)
            cols = [exp[:13].replace("_"," ")+":"+exp[13:15],exp[16:]]
            #for name,row in reader.items():
                #2021-04-02_0326_

            for m in print_efficiency_metrics:
                if m in reader:
                    if "values" in reader[m]:
                        cols.append(reader[m]["values"])
                    elif "time" in reader[m]:
                        cols.append(str(round(float(reader[m]["time"])/60,3))+" min")
                    elif "median_latency" in reader[m]:
                        cols.append(str(round(float(reader[m]["median_latency"])*1000,3))+" ms")
                    elif "sum" in reader[m]:
                        cols.append(str(round(float(reader[m]["sum"])/60,3))+" min")
                    else:
                        cols.append("-")
                else:
                    cols.append("-")

            table.add_row(*cols)

    console.print(table)
