from timeit import default_timer
import math
import statistics
import json
import numpy
from rich import box
from rich.console import Console
from rich.table import Table

def crappyhist(a, bins=20, width=30,range_=(0,1)):
    h, b = numpy.histogram(a, bins)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i], 
            '#'*int(width*h[i]/numpy.amax(h)), 
            h[i],#/len(a), 
            width=width))
    print('{:12.5f}  |'.format(b[bins]))


class PerformanceMonitor():

    instance = None
    @staticmethod
    def get():
        if PerformanceMonitor.instance == None:
            PerformanceMonitor.instance = PerformanceMonitor()
        return PerformanceMonitor.instance

    def __init__(self):
        self.timings = {}
        self.current_times = {}
        self.logs = {}
        self.unique_logs = {}
        self.gpu_info = {}
    
    # we assume that we only use the same gpu model if we do multi-gpu training
    def set_gpu_info(self,count,model):
        self.gpu_info["gpu_model"] = model
        self.gpu_info["gpu_count"] = count

    def start_block(self,category:str):
        self.current_times[category] = default_timer()

    def log_value(self,name:str,value):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(value)

    def log_unique_value(self,name:str,value):
        self.unique_logs[name] = value

    def stop_block(self,category:str,instances:int=1):
        if not category in self.timings:
            self.timings[category] = []

        self.timings[category].append((default_timer() - self.current_times[category], instances))

    def print_summary(self,console = None):
        if console == None:
            console = Console()

        table = Table(show_header=True, header_style="bold magenta")
        
        table.title = "Block-Timings"

        table.add_column("Block")
        table.add_column("Avg. it/s")
        table.add_column("Median Latency (ms)", justify="right")
        table.add_column("95th percentile latency (ms)", justify="right")
        table.add_column("Observations", justify="right")
        table.box = box.SIMPLE_HEAD

        for cat,data in self.timings.items():
            if len(data) == 1 and data[0][1] == 1:
                table.add_row(cat,"-", "{:.2f}".format(data[0][0]*1000),"-","1")
            else:
                #if len(data) > 1: # ignore the first as warm-up
                #    data = data[1:]
                per_iterations = [y/x for (x,y) in data]
                table.add_row(cat, "{:.2f}".format(statistics.median(per_iterations)),
                                   "{:.2f}".format(statistics.median([x*1000 for (x,_) in data])),
                                   "{:.2f}".format(numpy.percentile([x*1000 for (x,_) in data],95)),
                                   str(len(data)))
                #if cat == "search_nn_lookup":
                #    console.log("search_nn_lookup distribution")
                #    crappyhist([x*1000 for (x,_) in data],range_= (0,20))
        console.print(table)
        
        if len(self.unique_logs) > 0:
            table = Table(show_header=True, header_style="bold magenta")
            
            table.title = "Space / Memory Usage"
    
            table.add_column("Type")
            table.add_column("Space")
            table.box = box.SIMPLE_HEAD
    
            for cat,data in self.unique_logs.items():
                table.add_row(cat,str(data))
            
            console.print(table)

    def save_summary(self, file):
        
        summary = self.gpu_info.copy()
        total_gpu_hours = 0
        for cat,data in self.timings.items():
            cat_sum = {}
            times_only = [x for x,_ in data]
            if len(data) == 1 and data[0][1] == 1:
                cat_sum["type"] = "single_point"
                cat_sum["time"] = data[0][0]
            else:
                cat_sum["len"] = len(data)
                cat_sum["measure"] = "seconds"
                cat_sum["sum_times_gpus"] = sum(times_only) * self.gpu_info["gpu_count"]
                cat_sum["sum_gpu_hours"] = sum(times_only) * self.gpu_info["gpu_count"] / 60 / 60
                total_gpu_hours+=cat_sum["sum_gpu_hours"]
                cat_sum["sum"] = sum(times_only)

                #if len(data) > 2: # ignore the first as warm-up
                #    data = data[1:]
                per_iterations = [y/x for (x,y) in data]

                if len(per_iterations) > 1:
                    cat_sum["type"] = "list_iterations"
                    cat_sum["mean_perit"] = statistics.mean(per_iterations)
                    cat_sum["stdev_perit"] = statistics.stdev(per_iterations)
                    cat_sum["median_perit"] = statistics.median(per_iterations)
                    cat_sum["median_latency"] = statistics.median([x for (x,_) in data])
                    cat_sum["95th_latency"] = numpy.percentile([x for (x,_) in data],95)
                    cat_sum["observations"] = len(data)
                else:
                    cat_sum["type"] = "single_point_iterations"
                    cat_sum["perit"] = per_iterations

            summary[cat]=cat_sum

        for cat,data in self.logs.items():
            cat_sum = {}
            cat_sum["sum"] = statistics.mean(data) if len(data) > 1 else data[0]
            cat_sum["measure"] = "GB"
            summary[cat]=cat_sum

        for cat,data in self.unique_logs.items():
            cat_sum = {}
            cat_sum["values"] = data
            summary[cat]=cat_sum

        summary["total_gpu_hours"] = total_gpu_hours

        with open(file, "a") as out_file:
            json.dump(summary,out_file,indent=2)