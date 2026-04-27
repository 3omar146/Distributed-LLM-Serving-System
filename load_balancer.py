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
        worker = self.get_next_worker()
        return worker.process(request)