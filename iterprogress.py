import logging

class NoProgressBar(object):
    def __init__(self, *args, **kwargs): pass
    def start(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def update_i(self, *args, **kwargs): pass
    def finish(self, *args, **kwargs): pass
    def set_title(self, *args, **kwargs): pass

def with_progress(collection, length=None, title=None, pbar=NoProgressBar()):
    if length is None:
        length = len(collection)
    if title is not None:
        pbar.set_title(title)
    pbar.start(length)
    for elem in collection:
        yield elem
        pbar.update()

try:
    from progressbar import ProgressBar, Percentage, Bar, ETA, RotatingMarker
except ImportError:
    logging.warning(' progressbar package not installed. Progress cannot be '+
        'shown. See http://pypi.python.org/simple/progressbar or type '+
        '"sudo easy_install progressbar" to fix.')
    StandardProgressBar = NoProgressBar
else:
    class StandardProgressBar(object):
        def __init__(self, title='Progress: '):
            self.title = title
            self.is_finished = False

        def start(self, total, widgets=None):
            if widgets is None:
                widgets = [self.title, RotatingMarker(), ' ',
                            Percentage(), ' ', Bar(marker='='), ' ', ETA()]
            self.pbar = ProgressBar(widgets=widgets, maxval=total)
            self.pbar.start()
            self.i = 0

        def update(self, step=1):
            self.i += step
            self.pbar.update(self.i)
            if self.i == self.pbar.maxval:
                self.finish()

        def update_i(self, value):
            self.i = value
            self.pbar.update(value)
            if value == self.pbar.maxval:
                self.finish()

        def finish(self):
            if self.is_finished:
                pass
            else:
                self.pbar.finish()
                self.is_finished = True

        def set_title(self, title):
            self.title = title

