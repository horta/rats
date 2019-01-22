from matplotlib import pyplot as plt
import os
import sys
import numpy as np
from glob import glob
from copy import deepcopy
import seaborn as sns
import subprocess
import h5py
import pdb
from tqdm import tqdm
import pickle as pkl
from urllib.request import urlopen
from util import get_amelie_selection

root_left = sys.argv[1]
root_right = sys.argv[2]
root_output = sys.argv[3]
ext = sys.argv[4]

filenames_left = os.listdir(root_left)
filenames_right = os.listdir(root_right)

filenames_left = [f.replace("_qqplot", "") for f in filenames_left]
filenames_right = [f.replace("_manhattan", "") for f in filenames_right]

filenames = set(filenames_left).intersection(set(filenames_right))
try:
    os.makedirs(root_output)
except FileExistsError:
    pass

if os.path.exists("data/traits-kinship-hannah-amelie.pkl"):
    data = pkl.load(open("data/traits-kinship-hannah-amelie.pkl", "rb"))
else:
    url = "http://ipfs.io/ipfs/QmXtFuoA3JTkhhpUpqh4dyu3FJLUG7DWAmXwUk1N2TMbAh"
    data = pkl.load(urlopen(url))

def isbernoulli(x):
    x = np.asarray(x, float)
    u = np.unique(x)
    if len(u) == 2:
        return True
    return False

def get_bernoulli(x):
    x = np.asarray(x, float)
    u = np.unique(x)
    i0 = x == u[0]
    i1 = x == u[1]
    x[i0] = 0.0
    x[i1] = 1.0
    return x

def isdiscrete(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return all(x[ok] == np.asarray(x[ok], int))

def get_poisson(x):
    x = np.asarray(x, float)
    mi = min(x)    
    if mi < 0:
        x += -mi
    return x

def isnumber(x):
    try:
        np.asarray(x, float)
    except ValueError:
        return False
    return True

f = h5py.File("arrayexpress/HS.hdf5", "r")
samples = np.asarray(f["/imputed_genotypes/row_header/rat"].value)
samples = [i.decode() for i in samples]
f.close()

amelie_selection = get_amelie_selection()

data0 = deepcopy(data)
remove = []
for name in data0['measures'].columns.values:
    if isnumber(data0['measures'].loc[:, name]) and isdiscrete(data0['measures'].loc[:, name]):
        data0['measures'].loc[:, name] = get_poisson(data0['measures'].loc[:, name])
    else:
        remove.append(name)
for name in remove:
    del data0['measures'][name]

for filename in tqdm(filenames):
    dst_folder = root_output
    
    if ext == "png":
        left_filepath = os.path.join(root_left, filename).replace(f".{ext}", f"_qqplot.{ext}")
        right_filepath = os.path.join(root_right, filename).replace(f".{ext}", f"_manhattan.{ext}")
        output_filepath = os.path.join(root_output, filename)
#         print(["montage", left_filepath, right_filepath,
#                          "-tile", "1x2", "-geometry", "+0+0", output_filepath])
        subprocess.call(["montage", left_filepath, right_filepath,
                         "-tile", "1x2", "-geometry", "+0+0", output_filepath])
