import logging
import sys

class AppLogger:
    def __init__(self, debug, logger_name, log_filename = None):
        self.app_logger = logging.getLogger(logger_name)
        self.app_logger.propagate = False
        self.app_logger.setLevel(logging.DEBUG)

        self.console = logging.StreamHandler(sys.stdout)
        if debug:
            self.console.setLevel(logging.DEBUG)
        else:
            self.console.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)-8s %(message)s')
        self.console.setFormatter(formatter)
        self.app_logger.addHandler(self.console)

        if log_filename is not None:
            set_log_file(self.app_logger, log_filename)

    def get_logger(self):
        return self.app_logger

    def set_debug_console(self):
        self.console.setLevel(logging.DEBUG)

    def set_log_file(self, log_filename, regression=False):
        prim = logging.FileHandler(log_filename, 'a')
        prim.setLevel(logging.DEBUG)
        if regression:
            prim.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
        else:
            prim.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%y %H:%M'))
        self.app_logger.addHandler(prim)


