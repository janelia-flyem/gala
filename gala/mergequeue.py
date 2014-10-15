
from heapq import heapify, heappush, heappop
from iterprogress import NoProgressBar, StandardProgressBar

class MergeQueue(object):
    def __init__(self, items=[], length=None, with_progress=False, 
                                            prog_title='Agglomerating... '):
        if length is None:
            self.num_valid_items = len(items)
        else:
            self.num_valid_items = length
        self.original_length = self.num_valid_items
        self.q = items
        heapify(self.q)
        self.is_null_queue = len(items) == 0
        if with_progress:
            self.pbar = StandardProgressBar(prog_title)
        else:
            self.pbar = NoProgressBar()

    def __len__(self):
        return self.num_valid_items

    def finish(self):
        self.pbar.finish()

    def is_empty(self):
        return len(self.q) == 0

    def peek(self):
        while not self.q[0][1]:
            heappop(self.q)
        return self.q[0]

    def pop(self):
        self.pop = self.pop_no_start
        self.pbar.start(self.original_length)
        return self.pop_no_start()

    def pop_no_start(self):
        try:
            while not self.q[0][1]:
                _ = heappop(self.q)
        except IndexError:
            raise IndexError('pop from empty merge queue')
        item = heappop(self.q)
        self.invalidate(item)
        self.pbar.update_i(self.original_length - self.num_valid_items)
        return item

    def push(self, item):
        self.is_null_queue = False
        self.push = self.push_next
        self.push_next(item)

    def push_next(self, item):
        heappush(self.q, item)
        self.num_valid_items += 1

    def invalidate(self, item):
        if item[1]:
            self.num_valid_items -= 1
        item[1] = False

    def _total_len(self):
        return len(self.q)
