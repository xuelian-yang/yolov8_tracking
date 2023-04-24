# -*- coding: utf-8 -*-

import datetime
import logging
import os
import os.path as osp
from termcolor import colored


def d_print(text):
    print(colored(text, 'cyan'))

def d_print_r(text):
    print(colored(text, 'red'))

def d_print_g(text):
    print(colored(text, 'green'))

def d_print_b(text):
    print(colored(text, 'blue'))

def d_print_y(text):
    print(colored(text, 'yellow'))


def get_name(path):
    name, _ = osp.splitext(osp.basename(path))
    return name


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
