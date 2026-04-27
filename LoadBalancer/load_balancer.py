import threading

class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0
        self.lock = threading.Lock()

    def get_next_worker(self):
        with self.lock:
            worker = self.workers[self.index]
            self.index = (self.index + 1) % len(self.workers)
        return worker

    def dispatch(self, request):
        for _ in range(len(self.workers)): 
            worker = self.get_next_worker()
            try:
                return worker.process(request)
            except Exception:
                print(f"Request: {request.id} Worker {worker.id} failed, trying next...")
        raise Exception("All workers failed")