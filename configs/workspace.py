# -*- coding: utf-8 -*-

import os
import os.path as osp


class WorkSpace:
    def __init__(self):
        self.this = osp.dirname(__file__)
        self.home = osp.abspath(osp.join(self.this, '..'))
        self.temp = osp.join(self.home, 'temp')
        if not osp.exists(self.temp):
            os.makedirs(self.temp)
 
    def get_temp_dir(self):
        return self.temp
