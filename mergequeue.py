
from heapq import heapify, heappush, heappop
from iterprogress import NoProgressBar, StandardProgressBar
import logging
logging.basicConfig(filename='debug.txt', level=logging.DEBUG)

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
        if with_progress:
            self.pbar = StandardProgressBar(prog_title)
        else:
            self.pbar = NoProgressBar()

    def __len__(self):
        return len(self.q)

    def finish(self):
        self.pbar.finish()

    def peek(self):
        return self.q[0]

    def pop(self):
        self.pop = self.pop_no_start
        self.pbar.start(self.original_length)
        return self.pop_no_start()

    def pop_no_start(self):
        logging.debug('pop_no_start '+str(self.q[0]))
        if self.q[0][1]:
            self.num_valid_items -= 1
            logging.debug('  num_valid_items decremented '+
                str(self.num_valid_items)+' '+
                str(len([i for i in self.q if i[1]])-1))
            self.pbar.update_i(self.original_length - self.num_valid_items)
        return heappop(self.q)

    def push(self, item):
        heappush(self.q, item)
        self.num_valid_items += 1
        logging.debug('push '+str(item))
        logging.debug('  num_valid_items incremented '+
            str(self.num_valid_items)+' '+
            str(len([i for i in self.q if i[1]])))

    def invalidate(self, item):
        logging.debug('invalidate '+str(item))
        if item[1]:
            self.num_valid_items -= 1
            logging.debug('  num_valid_items decremented '+
                str(self.num_valid_items)+' '+
                str(len([i for i in self.q if i[1]])-1))
        item[1] = False
