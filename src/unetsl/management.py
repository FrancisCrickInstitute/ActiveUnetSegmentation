# -*- coding: utf-8 -*-
import pathlib
import queue, subprocess, threading, io, os, sys

import time

class GpuThread:
    def __init__(self, gpu, pool):
        self.gpu = gpu
        self.pool = pool
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.log = io.StringIO()
    def post(self, command):
        self.queue.put(command)
    def start(self):
        self.thread.start()
    def run(self):
        while True:
            cmd = self.queue.get()
            print("command recieved")
            if len(cmd)==0:
                break;
            else:
                self.log.write("\n### command: %s\n"%" ".join(cmd))
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"]=str(self.gpu)
                try:
                    start_time = time.time()
                    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, encoding="UTF8", env=env) as proc:
                        out, err = proc.communicate()
                        self.log.write("completed in: %f\n"%(time.time() - start_time))
                        self.log.write(out)
                        self.log.write("%error%")
                        self.log.write(err)
                        
                except:
                    self.log.write("%s \n\n %s \n\n %s"%sys.exc_info())
                finally:
                    self.pool.finished(self)
                
        print("finished worker thread")
    def save_log(self, out):
        self.log.seek(0)
        out.write(self.log.read())
        
class ProcessPool:
    def __init__(self):
        self._pool = queue.Queue()
        self.queue = queue.Queue()
        self.workers = []
        self.thread = threading.Thread(target=self.run)
        self._shutdown = False
        
    def shutdown(self):
        self._shutdown = True
        self.queue.put([])
        self.thread.join()
        for worker in self.workers:
            worker.thread.join()
        
    def post(self, command):
        if self._shutdown:
            raise Exception("no more tasks can be submitted. Shutdown!")
        self.queue.put(command);
    def addWorker(self, worker):
        self.workers.append(worker)
    def start(self):
        for worker in self.workers:
            worker.start()
            self._pool.put(worker)
        self.thread.start()
    def finished(self, worker):
        self._pool.put(worker)
        
    def run(self):
        while True:
            cmd = self.queue.get()
            if len(cmd)==0:
                break;
            worker = self._pool.get()
            worker.post(cmd)
        print("initializing worker shutdown")
        for worker in self.workers:
            worker.post([])
    def save_log(self, log_file):
        for worker in self.workers:
            worker.save_log(log_file)
            
def getGpuPool(numbers):
    pool = ProcessPool()
    
    for i in numbers:
        pool.addWorker(GpuThread(i, pool))
    
    pool.start()
    
    return pool

def getModels(model_folders):
    collect = []
    for model_folder in model_folders:
        collect += [f for f in pathlib.Path(model_folder).iterdir() if f.name.endswith("latest.h5")]
    return collect
    
    
def getOutputName(model_name, image_name, output_index):
    if "-csc" in model_name:
        return "pred%d-%s-%s"%(output_index, model_name, image_name);
    else:
        return "pred-%s-%s"%(model_name.replace(".h5", ""), image_name)
