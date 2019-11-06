from timeit import default_timer
import math
import statistics
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

    def start_block(self,category:str):
        self.current_times[category] = default_timer()

    def log_value(self,name:str,value):
        self.logs[name] = value

    def stop_block(self,category:str,instances:int=1):
        if not category in self.timings:
            self.timings[category] = []

        self.timings[category].append((default_timer() - self.current_times[category], instances))

    def print_summary(self):
        for cat,data in self.timings.items():
            if len(data) == 1 and data[0][1] == 1:
                print(cat, data[0][0], "s")
            else:
                if len(data) > 1: # ignore the first as warm-up
                    data = data[1:]
                per_iterations = [y/x for (x,y) in data]
                print(cat, statistics.mean(per_iterations), "it/s")
        for cat,data in self.logs.items():
            print(cat,data)
    def save_summary(self, file):
        with open(file, "w") as out_file:
            out_file.write("cat\tmean\tstddev\tmedian\tobservations")
            for cat,data in self.timings.items():
                if len(data) == 1 and data[0][1] == 1:
                    out_file.write(cat  +"\t"+ str(data[0][0]) +"s\n")
                else:
                    if len(data) > 2: # ignore the first as warm-up
                        data = data[1:]
                    per_iterations = [y/x for (x,y) in data]
                    out_file.write(cat +"\t"+ str(statistics.mean(per_iterations))+"\t"+ str(statistics.stdev(per_iterations))+"\t"+str(statistics.median(per_iterations))+ "it/s\t"+str(len(per_iterations))+"\n")
            for cat,data in self.logs.items():
                out_file.write(cat+"\t"+ str(data)+"\n")