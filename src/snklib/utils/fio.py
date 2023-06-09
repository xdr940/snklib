import json
import pandas as pd
import datetime
from path import Path
def json2dict(file):
    with open(file, 'r') as f:
        dict = json.load(fp=f)
        return dict
def dict2json(file,dict):
    with open(file, 'w') as f:
        json.dump(dict, f)


def to_csv(file,series):
    series.to_csv(file)
def read_csv(file):
    return pd.read_csv(file,index_col=0)

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def readtles(filename):
    lines = readlines(filename)
    length = len(lines)
    assert(length%3==1,"err")


    reformat_lines=[]
    cnt = 1
    while cnt+3<= length:
        satname,line1,line2 = lines[cnt],lines[cnt+1],lines[cnt+2]
        cnt+=3
        reformat_lines.append([satname,line1,line2])

    return reformat_lines

def mkdir_with_datetime(save_path):
    yy = datetime.datetime.now().year
    mm = datetime.datetime.now().month
    dd = datetime.datetime.now().day
    hh = datetime.datetime.now().hour
    mi = datetime.datetime.now().minute
    ss = datetime.datetime.now().second
    stem = "{}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(yy, mm, dd, hh, mi, ss)
    save_dir = Path(save_path) / stem
    save_dir.mkdir_p()
    return save_dir