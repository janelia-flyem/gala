import logging
import sys

def set_logger(debug, logger_name, log_filename = None):
    app_logger = logging.getLogger(logger_name)
    app_logger.propagate = False
    app_logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    if debug:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    app_logger.addHandler(console)

    if log_filename is not None:
        set_log_file(app_logger, log_filename)

    return app_logger

def set_debug_console():
    console = logging.StreamHandler(sys.stdout)
    console.setlevel(logging.DEBUG)

def set_log_file(app_logger, log_filename):
    prim = logging.FileHandler(log_filename, 'a')
    prim.setLevel(logging.DEBUG)
    prim.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d-%y %H:%M'))
    app_logger.addHandler(prim)


