import logging
import multiprocessing as mtp
import queue
import random
import string
import time
from abc import ABCMeta, abstractmethod
from ast import arg
from threading import Thread

import numpy as np


class ComputeEngineClient():
    
    def __init__(self, server_queues, client_queue):
        self._LOG = logging.getLogger(self.__class__.__name__)
        self.client_id = client_queue.id
        self.server_queues = server_queues
        self.client_queue = client_queue
        self.queue_id = 0
        self.ids_pool = np.random.permutation(np.arange(len(server_queues)))
        self.counter = 0
        
    def request(self, in_data, block=True):
        self.counter = (self.counter + 1)%len(self.ids_pool)
        self.queue_id = self.ids_pool[self.counter]
        queue = self.server_queues[self.queue_id]
        queue.put(self.client_id, in_data, block=block)
        client_id,out_data = self.client_queue.get(block=block)
        return out_data
    
class CommQueue(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, id): pass
    
    @abstractmethod
    def get(self, block=True): pass
    
    @abstractmethod
    def put(self, request_id, reques_data, block=True): pass
    
    @abstractmethod
    def empty(self): pass
    
    
class MTPCommQueue(CommQueue):
    
    def __init__(self, id):
        self.id = id
        self.q = mtp.Queue()
        
    def get(self, block=True):
        return self.q.get(block=block)

    def put(self, request_id, request_data, block=True):
        return self.q.put([request_id, request_data], block=block)
    
    def empty(self):
        return self.q.empty()
    
class ComputeEngineManager():
    
    def __init__(self, compute_engines, num_clients=1, batch_size=1, comm_queue_class=CommQueue, session_prefix=None):
        self._LOG = logging.getLogger(self.__class__.__name__)
        server_ids, server_queues, server_processes = [], [], []
        if session_prefix is None: session_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
        client_ids = [ "{}_client_{}".format(session_prefix, i) for i in range(num_clients) ]
        client_queues = { client_id: comm_queue_class(client_id) for client_id in client_ids }
        for ce_idx, compute_engine in enumerate(compute_engines):
            server_ids += [ "{}_server_{}".format(session_prefix, ce_idx) ]
            server_queue = comm_queue_class( server_ids[-1] )
            server_process = mtp.Process(target=self.inference_manager, args=[ compute_engine, server_queue, client_queues, batch_size ]) 
            server_process.daemon = True  
            server_queues.append(server_queue)
            server_processes.append(server_process)
        clients = [ ComputeEngineClient( server_queues, client_queues[client_id]) for client_id in client_ids ]
        self.clients = clients
        self.server_processes = server_processes
        self.batch_size = batch_size
        self.server_ids = server_ids
        self.client_ids = client_ids
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
                    client_queues[client_id].put(client_id,out_data)
            
class ComputeEngine(metaclass=ABCMeta):
    
    @abstractmethod
    def load(self): pass
    
    @abstractmethod
    def process(self, data:list): pass
    
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
    compute_engine = ComputeEngineManager(list_of_ce, num_clients=3, batch_size=1, comm_queue_class=MTPCommQueue)
    compute_engine.start()
    clients = compute_engine.get_clients()
    
    t0 = time.time()
    print("============= SINGLE PROCESS =============")    
    # sequential
    for inp in range(100):
        out = clients[inp%len(clients)].request(inp)
        print("input: {}, output: {}, =============> success: {}".format(inp, out, inp==out)) 
    tT = time.time()
    print('============= SINGLE PROCESS ============= END: {} / {} req/sec'.format(100, tT-t0))
    
    print("============= MULTI THREAD =============")  
    t0 = time.time()     
    client_threads = [ Thread(target=test_client, args=[client]) for client in clients ]
    for client_thread in client_threads: client_thread.start()     
    for client_thread in client_threads: client_thread.join()
    tT = time.time()
    print('============= MULTI THREAD ============= END: {} / {} req/sec'.format(100*3, tT-t0))
    
    print("============= MULTI PROCESS =============")       
    t0 = time.time() 
    client_processes = [ mtp.Process(target=test_client, args=[client]) for client in clients ]
    for client_process in client_processes: client_process.start()     
    for client_process in client_processes: client_process.join()
    tT = time.time()
    print('============= MULTI PROCESS ============= END: {} / {} req/sec'.format(100*3, tT-t0))   