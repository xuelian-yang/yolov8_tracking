# -*- coding: utf-8 -*-

import contextlib
import datetime
import logging
import os
import os.path as osp
from termcolor import colored
import time


def d_print(text):
    print(colored(text, 'cyan'))


def setup_log(filename):
    medium_format = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )
    if not filename.lower().endswith('.log'):
        filename = filename + '.log'
    log_dir = osp.abspath(osp.join(osp.dirname(__file__), '../logs'))
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    dt_now = datetime.datetime.now()
    get_log_file = osp.join(log_dir, filename)
    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print(colored('@{} created at {}'.format(get_log_file, dt_now), 'magenta'))


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
    """

    def __init__(self, name="unknown", t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.name = name
        self.t = t

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        return time.time()
