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


class ComputeEngine(metaclass=ABCMeta):
    """Abstract class for the compute engine"""
    
    @abstractmethod
    def load(self): pass
    
    @abstractmethod
    def process(self, data:list): pass
    
class CommQueue(metaclass=ABCMeta):
    """Abstract class for the communication queue. A communication queue is used between a client and a compute engine (server)"""

    @abstractmethod
    def __init__(self, id): pass
    
    @abstractmethod
    def get(self): pass
    
    @abstractmethod
    def put(self, request_id, reques_data): pass
    
    @abstractmethod
    def empty(self): pass


class ComputeEngineClient():
    
    """compute engine client communicating with compute engines"""
    
    def __init__(self, server_queues, client_queue):
        self._LOG = logging.getLogger(self.__class__.__name__)
        self.client_id = client_queue.id
        self.server_queues = server_queues
        self.client_queue = client_queue
        self.queue_id = 0
        self.ids_pool = np.random.permutation(np.arange(len(server_queues)))
        self.counter = 0
        
    def request(self, in_data):
        self.counter = (self.counter + 1)%len(self.ids_pool)
        self.queue_id = self.ids_pool[self.counter]
        queue = self.server_queues[self.queue_id]
        queue.put(self.client_id, in_data)
        client_id,out_data = self.client_queue.get()
        return out_data
    
class ComputeEngineManager():
    
    """compute engine manager puts in communication compute engines(servers) and clients via communication queues"""
    
    def __init__(self, compute_engines, num_clients=1, batch_size=1, comm_queue_class=CommQueue, session_prefix=None):
        
        self._LOG = logging.getLogger(self.__class__.__name__)
        self.num_compute_engines = len(compute_engines)
        self.num_clients = num_clients
        if session_prefix is None: 
            session_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            self._LOG.info("session prefix not defined one will be created: {}".format(session_prefix))
        self.session_prefix = session_prefix
        self._LOG.info("initializeing identifiers for clients and server")
        self.initialize_ids()
        self.batch_size = batch_size
        self.comm_queue_class = comm_queue_class
        self.compute_engines = compute_engines
        self._LOG.info("initialize_servers")
        self.initialize_servers()
        self._LOG.info("initialize_clients")
        self.initialize_clients()
    
    def initialize_servers(self):
        compute_engines = self.compute_engines
        batch_size = self.batch_size
        comm_queue_class = self.comm_queue_class
        server_queues, server_processes = [], []
        client_ids = self.client_ids
        server_ids = self.server_ids
        # create the queues to be passed to each compute engine
        # each compute engine is assigned a Process in wich the inference_manager method will run
        # each compute engins is also assigned a comunitacion queue to listen to
        client_queues = { client_id: comm_queue_class(client_id) for client_id in client_ids }
        for ce_idx, compute_engine in enumerate(compute_engines):
            server_queue = comm_queue_class( server_ids[ce_idx] )
            server_process = mtp.Process(target=self.inference_manager, args=[ compute_engine, server_queue, client_queues, batch_size ]) 
            server_process.daemon = True  
            server_queues.append(server_queue)
            server_processes.append(server_process)
        self.client_queues = client_queues
        self.server_queues = server_queues
        self.server_processes = server_processes
            
    def initialize_clients(self):
        client_ids = self.client_ids
        client_queues = self.client_queues
        server_queues = self.server_queues
        clients = [ ComputeEngineClient( server_queues, client_queues[client_id]) for client_id in client_ids ]
        self.clients = clients
        
    def initialize_ids(self):
        self.client_ids = [ "{}_client_{}".format(self.session_prefix, i) for i in range(self.num_clients) ]
        self.server_ids = [ "{}_server_{}".format(self.session_prefix, i) for i in range(self.num_compute_engines) ]
            
    def start(self):
        self._LOG.info("starting servers")
        for server_process in self.server_processes: server_process.start()
            
    def get_clients(self): return self.clients        
        
    @staticmethod
    def inference_manager(compute_engine, server_queue, client_queues, batch_size):
        """main method for the compute engine, an infinite loop that listes to data in the communication queue and replies to clients
        Args:
            compute_engine (ComputeEngine): the main compute engine implementing the load and process method
            server_queue (CommQueue): the queue onto wich clients send their requests
            client_queues (CommQueue dict): dictionary containing communication queues to reply to clients
            batch_size (int): batching request from different clients
        """
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
    
# TEST IMPLEMENTATION
    
class TestCE(ComputeEngine):
    
    def load(self): pass
    
    def process(self, data): 
        time.sleep(0.01)
        return data
    
class MTPCommQueue(CommQueue):
    
    def __init__(self, id):
        self.id = id
        self.q = mtp.Queue()
        
    def get(self):
        return self.q.get(block=True)

    def put(self, request_id, request_data):
        return self.q.put([request_id, request_data], block=True)
    
    def empty(self):
        return self.q.empty()
    
def test_client(client):
    for inp in range(3):
        out = client.request(inp)
        print("input: {}, output: {}, =============> success: {} ============> client_id: {}".format(inp, out, inp==out, client.client_id)) 
    
if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',level=logging.INFO)
    
    list_of_ce = [ TestCE(), TestCE(), TestCE() ]
    compute_engine = ComputeEngineManager(list_of_ce, num_clients=10, batch_size=1, comm_queue_class=MTPCommQueue)
    compute_engine.start()
    clients = compute_engine.get_clients()
    
    t0 = time.time()
    print("============= SINGLE PROCESS =============")    
    # sequential
    for inp in range(3):
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
