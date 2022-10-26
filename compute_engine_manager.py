from ast import arg
import multiprocessing as mtp
import queue
import random
import logging
import string
from threading import Thread
import time
from abc import ABCMeta, abstractmethod

class ComputeEngineClient():
    
    def __init__(self, client_id, server_queues, client_queue):
        self._LOG = logging.getLogger(self.__class__.__name__)
        self.client_id = client_id
        self.server_queues = server_queues
        self.client_queue = client_queue
        self.queue_id = 0
        
    def request(self, in_data, block=True):
        self.queue_id = random.randint(0, len(self.server_queues)-1)
        queue = self.server_queues[self.queue_id]
        queue.put([self.client_id, in_data], block=block)
        out_data = self.client_queue.get(block=block)
        return out_data
    
class ComputeEngineManager():
    
    def __init__(self, compute_engines, num_clients=1, batch_size=1):
        self._LOG = logging.getLogger(self.__class__.__name__)
        client_queues, server_queues, server_processes = [], [], []
        for i in range(num_clients): client_queues.append( mtp.Queue() )
        for ce_idx, compute_engine in enumerate(compute_engines):
            server_queue = mtp.Queue()
            server_process = mtp.Process(target=self.inference_manager, args=[ compute_engine, server_queue, client_queues, batch_size ]) 
            server_process.daemon = True  
            server_queues.append(server_queue)
            server_processes.append(server_process)        
        clients = []
        for i in range(num_clients): clients.append(ComputeEngineClient( i, server_queues, client_queues[i] ))
        self.clients = clients
        self.server_processes = server_processes
        self.batch_size = batch_size
        self._LOG.info("instantiation of servers and clients")
            
    def start(self):
        self._LOG.info("startimg servers")
        for server_process in self.server_processes: server_process.start()
            
    def get_clients(self): return self.clients        
        
    @staticmethod
    def inference_manager(compute_engine, server_queue, client_queues, batch_size):
        compute_engine.load()
        while True:
            client_ids, in_datas = [], []
            for i in range(batch_size):
                if server_queue.empty(): break
                client_id, in_data = server_queue.get()
                client_ids.append(client_id)
                in_datas.append(in_data)
                
            if len(in_datas)>0:
                out_datas = compute_engine.process(in_datas)
                for client_id, out_data in zip(client_ids, out_datas):
                    client_queues[client_id].put(out_data)
            
class ComputeEngine(metaclass=ABCMeta):
    
    @abstractmethod
    def load(self): pass
    
    @abstractmethod
    def process(self, client_id, data): pass
    
class TestCE(ComputeEngine):
    
    def load(self): pass
    
    def process(self, data): 
        time.sleep(0.01)
        return data
    
def test_client(client):
    for inp in range(100):
        out = client.request(inp)
        print("input: {}, output: {}, =============> success: {} ============> client_id: {}".format(inp, out, inp==out, client.client_id)) 
    
if __name__ == "__main__":
    
    list_of_ce = [ TestCE(), TestCE(), TestCE() ]
    compute_engine = ComputeEngineManager(list_of_ce, num_clients=3, batch_size=2)
    compute_engine.start()
    clients = compute_engine.get_clients()
    
    print("============= SINGLE PROCESS =============")    
    # sequential
    for inp in range(100):
        out = clients[inp%len(clients)].request(inp)
        print("input: {}, output: {}, =============> success: {}".format(inp, out, inp==out)) 
    
    print("============= MULIT THREAD =============")       
    client_threads = [ Thread(target=test_client, args=[client]) for client in clients ]
    for client_thread in client_threads: client_thread.start()     
    for client_thread in client_threads: client_thread.join()
    
    print("============= MULTI PROCESS =============")        
    client_processes = [ mtp.Process(target=test_client, args=[client]) for client in clients ]
    for client_process in client_processes: client_process.start()     
    for client_process in client_processes: client_process.join()
            