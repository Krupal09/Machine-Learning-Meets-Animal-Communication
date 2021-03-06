"""
Module: logging.py
Authors: Christian Bergler, Hendrik Schroeter
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 12.12.2019
"""

import logging # https://docs.python.org/3/howto/logging.html
import logging.handlers
import os
import queue


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    """
    Logger in order to track training, validation, and testing of the network
    """

    def __init__(self, name, debug=False, log_dir=None, do_log_name=False):
        level = logging.DEBUG if debug else logging.INFO
        fmt = "%(asctime)s"
        if do_log_name:
            fmt += "|%(name)s"
        fmt += "|%(levelname).1s|%(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers = [sh]

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fs = logging.FileHandler(os.path.join(log_dir, name + ".log"))
            fs.setFormatter(formatter)
            handlers.append(fs)

        self._queue = queue.Queue(1000)
        self._handler = logging.handlers.QueueHandler(self._queue)
        self._listener = logging.handlers.QueueListener(self._queue, *handlers)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addHandler(self._handler)
        self._listener.start()

    def close(self):
        self._listener.stop()
        self._handler.close()

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    """Rachael: I don't think the below method is used in the whole pipeline. Otherwise, uncomment the code."""
    #def epoch(
    #    self,
    #    phase,
    #    epoch,
    #    num_epochs,
    #    loss,
    #    accuracy=None,
    #    f1_score=None,
    #    precision=None,
    #    recall=None,
    #    lr=None,
    #    epoch_time=None,
    #):
    #    s = "{}|{:03d}/{:d}|loss:{:0.3f}".format(
    #        phase.upper().rjust(5, " "), epoch, num_epochs, loss
    #    )
    #    if accuracy is not None:
    #        s += "|acc:{:0.3f}".format(accuracy)
    #    if f1_score is not None:
    #        s += "|f1:{:0.3f}".format(f1_score)
    #    if precision is not None:
    #        s += "|pr:{:0.3f}".format(precision)
    #    if recall is not None:
    #        s += "|re:{:0.3f}".format(recall)
    #    if lr is not None:
    #        s += "|lr:{:0.2e}".format(lr)
    #    if epoch_time is not None:
    #        s += "|t:{:0.1f}".format(epoch_time)

    #    self._logger.info(s)
