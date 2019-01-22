import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import limix
import pickle as pkl
import numpy as np
import requests
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
import h5py
import re
from plot import plot_poisson_ordered, plot_distplots, plot_distplot
import h5py
import dask
import dask.dataframe
import dask.array
from adjustText import adjust_text
from urllib.request import urlopen
from joblib import Parallel, delayed
import gzip
from veja import bokeh_manhattan

only_hash = True
data = pkl.load(open("data/traits-kinship-hannah-amelie.pkl", "rb"))
DST_FOLDER = "3"

def get_amelie_selection():
    return list(pd.read_csv("amelie trait selection.txt", header=None).values.ravel())

dst_folder = "3"
if not os.path.exists(dst_folder):
    os.mkdir(dst_folder)
if not os.path.exists(join(dst_folder, 'phenotype')):
    os.mkdir(join(dst_folder, 'phenotype'))
if not os.path.exists(join(dst_folder, 'kinship')):
    os.mkdir(join(dst_folder, 'kinship'))
    
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

f = h5py.File("arrayexpress/HS_shuf.hdf5", "r")
samples = np.asarray(f["/imputed_genotypes/row_header/rat"].value)
samples = [i.decode() for i in samples]
f.close()

data0 = deepcopy(data)
remove = []
for name in data0['measures'].columns.values:
    ok0 = isnumber(data0['measures'].loc[:, name])
    if ok0 and isdiscrete(data0['measures'].loc[:, name]):
        data0['measures'].loc[:, name] = get_poisson(data0['measures'].loc[:, name])
    else:
        remove.append(name)
for name in remove:
    del data0['measures'][name]

f = h5py.File("arrayexpress/HS_shuf.hdf5", "r")
pos = []
chrs = []
for i in tqdm(range(1, 22)):
    p = f["/imputed_genotypes/chr{}/col_header/pos".format(i)].value
    pos.append(p)
    chrs.append([i] * len(p))
f.close()

pos = np.concatenate(pos).astype(float)[np.load("maf.npy") >= 0.05]
chrs = np.concatenate(chrs).astype(int)[np.load("maf.npy") >= 0.05]

amelie_selection = get_amelie_selection()

patts = ["scan_measures_normal_(.*)_quantile_gaussianize.json.pkl",
         "scan_measures_poisson_(.*).json.pkl"]

patt2name = {'scan_measures_poisson_(.*).json.pkl':'MPoisson',
             'scan_measures_normal_(.*)_quantile_gaussianize.json.pkl':'MNormalGau'}

def plot_this(d, test, debug=False):
    alpha = 0.01
    trait = d[0]["trait"]
    pos = d[0]["pos"]
    chrs = d[0]["chrs"]
    amelie_selection = d[0]["amelie_selection"]
    
    pos = np.asarray(pos)[chrs != 21]
    chrs = np.asarray(chrs)[chrs != 21]
    
    data = {}
    pv_min = np.inf
    for i, di in enumerate(d):
        
        with gzip.open(open(di["path"], "rb")) as f:
            data[di["model"]] = pkl.load(f)
        
        data[di["model"]]["pv"] = np.asarray(data[di["model"]]["pv"])[:len(chrs)]
        
        if test != "bonferroni":
            r = limix.stats.multipletests(data[di["model"]]["pv"], alpha=alpha, method=test)
            data[di["model"]]["pv"] = r[1]
        
        pv_min = min(pv_min, np.min(np.asarray(data[di["model"]]["pv"])))
    
    log_pv_min = -np.log10(pv_min)
    lpv_max = np.ceil(np.ceil(log_pv_min) + np.ceil(log_pv_min) * 0.05)
    
    if debug:
        pdb.set_trace()

    for i, di in enumerate(d):
        pv = data[di["model"]]["pv"]
        trait = di["trait"]
        model = di["model"]

        dataframe = {"pv": pv, "pos": pos, "chr": chrs}

        df = pd.DataFrame(data=dataframe)
        
        plt.figure(figsize=(20, 6))
        limix.plot.manhattan(df)
        if test == "bonferroni":
            plt.axhline(-np.log10(alpha / df.shape[0]), color='red')
        else:
            plt.axhline(-np.log10(alpha), color='red')
        plt.axhline(log_pv_min, color='black', lw=0.5, zorder=-20, alpha=0.80)

        txts = []
        nsnps = df.shape[0]
        ax = plt.gca()
        txt = ax.text(
            0.20,
            0.90,
            f"#snps={nsnps}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        txts.append(txt)

        txt = ax.text(
            0.20,
            0.85,
            "pv={:e}".format(np.min(df["pv"])),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        txts.append(txt)
        
        txt = ax.text(
            0.20,
            0.80,
            "$\\alpha={:.2f}$".format(alpha),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        txts.append(txt)

        
        plt.ylim([0, lpv_max])
        ax = plt.gca()

        if trait in amelie_selection:
            ax.set_title(f"{trait} - {model}", color='red')
        else:
            ax.set_title(f"{trait} - {model}")
        folder = f"3/fig/scan/{model}/{trait}"

        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
        adjust_text(txts, autoalign="y", only_move={"text": "y"})
        print("Saving files...")
        if test != "bonferroni":
            print("Saving png...")
            plt.savefig(folder + f"/manhattan_{test}.png", bbox_inches='tight')
            print("Saving pdf...")
            plt.savefig(folder + f"/manhattan_{test}.pdf", bbox_inches='tight')
            print("Saving html...")
            bokeh_manhattan(df, folder + f"/manhattan_{test}.html")
        else:
            print("Saving png...")
            plt.savefig(folder + "/manhattan.png", bbox_inches='tight')
            print("Saving pdf...")
            plt.savefig(folder + "/manhattan.pdf", bbox_inches='tight')
            print("Saving html...")
            bokeh_manhattan(df, folder + f"/manhattan.html")
        plt.close()

args = []
data = dict()
for path in glob(join(dst_folder, "*.json.pkl.gz")):
    filename = path.split("/")[1]
    for patt in patts:
        match = re.match(patt, filename)
        if match:
            trait = match.groups(0)[0]
            model = patt2name[patt]
            if trait not in data:
                data[trait] = []

            data[trait].append({"trait": trait, "model": model, "path": path, "pos": pos,
                                "chrs":chrs, "amelie_selection":amelie_selection})

# Number of seeds
N = int(sys.argv[1])

# Seed id
seed = int(sys.argv[2])

# Shall we debug?
debug = False
test = "bonferroni"
if len(sys.argv) > 3:
    if "--debug" in sys.argv[2:]:
        DEBUG = True
    if "--fdr_bh" in sys.argv[2:]:
        test = "fdr_bh"

for (i, a) in enumerate(list(data.values())):
    if i % N == seed:
        plot_this(a, test, debug)
