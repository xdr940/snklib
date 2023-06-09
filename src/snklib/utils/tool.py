
import time
import numpy as np
import random
import pandas as pd
import math

def typeof(variate):
    type=None
    if isinstance(variate,int):
        type = "int"
    elif isinstance(variate,str):
        type = "str"
    elif isinstance(variate,float):
        type = "float"
    elif isinstance(variate,list):
        type = "list"
    elif isinstance(variate,tuple):
        type = "tuple"
    elif isinstance(variate,dict):
        type = "dict"
    elif isinstance(variate,set):
        type = "set"
    return type


def time_stat(last_end):
    end = time.time()
    running_time = end - last_end
    # print('--> time cost : %.5f sec' % running_time)
    return running_time

def get_now():
    return time.time()

def list_filter(lines,perc=0.1):
    random.shuffle(lines)
    length = len(lines)
    length = int(length*perc)
    return lines[:length]



def RandomPath(paths):
    num_paths = len(paths)
    if num_paths==0:
        return [],None
    else:
        max_ids =num_paths
        id = random.randint(0,max_ids-1)
        return paths[id],id

def LHpath(paths):
    minlen = 1000
    path_ret = []
    idx_ret=0
    for idx,path in enumerate( paths):
        if len(path_ret) < minlen:
            minlen = len(path)
            path_ret = path
            idx_ret=idx
    return path_ret,idx_ret
