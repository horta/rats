import pdb
import matplotlib
matplotlib.use('Agg')

font = {'size'   : 24}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)
import gzip

from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb
import limix
import sys
import pickle as pkl
import numpy as np
import seaborn as sns
from numpy import asarray
import xarray as xr
from copy import deepcopy
import os
from os.path import join
import simplejson as json
import hashlib
import pandas as pd
import pdb
from glob import glob
from joblib import Parallel, delayed
import h5py
import re
from plot import plot_poisson_ordered, plot_distplots, plot_distplot
import h5py
import dask
import dask.dataframe
import dask.array
from urllib.request import urlopen

only_hash = True
data = pkl.load(open("data/traits-kinship-hannah-amelie.pkl", "rb"))
DST_FOLDER = "3"

def get_amelie_selection():
    return list(pd.read_csv("amelie trait selection.txt", header=None).values.ravel())

if not os.path.exists(DST_FOLDER):
    os.mkdir(DST_FOLDER)
if not os.path.exists(join(DST_FOLDER, 'phenotype')):
    os.mkdir(join(DST_FOLDER, 'phenotype'))
if not os.path.exists(join(DST_FOLDER, 'kinship')):
    os.mkdir(join(DST_FOLDER, 'kinship'))


def is_discrete(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return all(x[ok] == np.asarray(x[ok], int))

def get_poisson(x):
    x = np.asarray(x, float)
    mi = min(x)    
    if mi < 0:
        x += -mi
    return x

def is_number(x):
    try:
        np.asarray(x, float)
    except ValueError:
        return False
    return True

with h5py.File("arrayexpress/HS_shuf.hdf5", "r") as f:
    samples = np.asarray(f["/imputed_genotypes/row_header/rat"].value)
    samples = [i.decode() for i in samples]
    
with h5py.File("arrayexpress/HS_shuf.hdf5", "r") as f:
    pos = []
    chrs = []
    for i in range(1, 22):
        p = f["/imputed_genotypes/chr{}/col_header/pos".format(i)].value
        pos.append(p)
        chrs.append([i] * len(p))

pos = np.concatenate(pos).astype(float)[np.load("maf.npy") >= 0.05]
chrs = np.concatenate(chrs).astype(int)[np.load("maf.npy") >= 0.05]
amelie_selection = get_amelie_selection()

patts = ["null_measures_normal_(.*)_quantile_gaussianize.json.pkl",
         "null_measures_poisson_(.*).json.pkl"]

patt2name = {'null_measures_poisson_(.*).json.pkl':'MPoisson',
             'null_measures_normal_(.*)_mean_standardize.json.pkl':'MNormalStd',
             'null_measures_normal_(.*)_quantile_gaussianize.json.pkl':'MNormalGau'}

def plot_trait(d, debug=False):
    if debug:
        pdb.set_trace()

    pos = list(d[0]["pos"]) * 5
    chrs = list(d[0]["chrs"]) * 5
    amelie_selection = d[0]["amelie_selection"]
    
    # Load results
    data = {}
    for i, di in enumerate(d):
        with gzip.open(di["path"], 'rb') as f:
            data[di["model"]] = pkl.load(f)

    for i, di in enumerate(d):

        pv = data[di["model"]]["pv"]
        if len(pv) != len(pos):
            print("len(pv) != len(pos)")
            return None

        trait = di["trait"]
        model = di["model"]
        
        df = pd.DataFrame(data={"pv": pv, "pos": pos, "chr": chrs})
        plt.figure(figsize=(6, 6))
                
        limix.plot.qqplot(df[df["chr"] != 21]["pv"], label="chrom 1-20")
        limix.plot.qqplot(df[df["chr"] == 21]["pv"], label="chrom 21")

        ax = plt.gca()
        ax.legend(loc='lower right')
        # Color in RED Amelie selection
        if trait in amelie_selection:
            ax.set_title(f"{trait} - {model}", color='red')
        else:
            ax.set_title(f"{trait} - {model}")

        try:
            folder = f"{DST_FOLDER}/fig/null/{model}/{trait}"
            os.makedirs(folder)
        except FileExistsError:
            pass
        
        fig = plt.gcf()
        fig.savefig(folder + "/qqplot.png", bbox_inches='tight')
        fig.savefig(folder + "/qqplot.pdf", bbox_inches='tight')
        plt.close()

args = []
data = dict()
for path in glob(join(DST_FOLDER, "*.json.pkl.gz")):
    filename = path.split("/")[1]
    for patt in patts:
        match = re.match(patt, filename)
        if match:
            trait = match.groups(0)[0]
            model = patt2name[patt]
            if trait not in data:
                data[trait] = []

            data[trait].append({"trait": trait, "model": model, "path": path, "pos": pos,
                                "chrs": chrs, "amelie_selection": amelie_selection})

# Number of seeds
N = int(sys.argv[1])

# Seed id
seed = int(sys.argv[2])

# Shall we debug?
debug = False
if len(sys.argv) > 3:
    if "--debug" in sys.argv[2:]:
        debug = True

for (i, a) in enumerate(list(data.values())):
    if i % N == seed:
        plot_trait(a, debug)