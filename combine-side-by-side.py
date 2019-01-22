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

folders_left = os.listdir(root_left)
folders_right = os.listdir(root_right)

folders = set(folders_left).intersection(set(folders_right))
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
    
for folder in tqdm(folders):
    dst_folder = os.path.join(root_output, folder)
    
    left_filenames = [str(os.path.basename(f)) for f in glob(os.path.join(root_left, folder, f"*.{ext}"))]
    right_filenames = [str(os.path.basename(f)) for f in glob(os.path.join(root_right, folder, f"*.{ext}"))]
    
    for left_filename in left_filenames:
    
        filenames = list(set([left_filename]).intersection(set(right_filenames)))
        if len(filenames) == 0:
            continue

        for filename in filenames:

            left_filepath = os.path.join(root_left, folder, filename)
            right_filepath = os.path.join(root_right, folder, filename)
            dirname = os.path.dirname(folder)
            basename = os.path.basename(folder)
            output_filepath = os.path.join(root_output, dirname, basename + "_" + filename)

            plt.figure(figsize=(6, 6))
            series = data0['measures'][basename]
            y = series.astype(float).dropna().values
            sns.distplot(y)
            ax = plt.gca()
            ax.set_xlabel(basename)

            if basename in amelie_selection:
                ax.xaxis.label.set_color('red')

            pheno_filepath = os.path.join(root_output, dirname, basename + f"_pheno.{ext}")
            plt.savefig(pheno_filepath)
            plt.close()

            if ext == "png":
                subprocess.call(["montage", pheno_filepath, left_filepath, right_filepath,
                                 "-tile", "3x1", "-geometry", "+0+0", output_filepath])
            elif ext == "pdf":
                cmd = f"latexdockercmd.sh pdfjam --nup 2x1 {left_filepath} {right_filepath} --outfile {output_filepath}"
                print(cmd)

            os.remove(pheno_filepath)
