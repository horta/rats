import random
from tqdm import tqdm
import sys
import pdb
import limix
import pickle as pkl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
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
from urllib.request import urlopen
try:
    import ipfsapi
    has_ipfs = True
except ModuleNotFoundError:
    has_ipfs = False


if has_ipfs:
    try:
        ipfs = ipfsapi.connect('127.0.0.1', 5001)
    except Exception:
        has_ipfs = False

only_hash = True
MAF = 0.01


if os.path.exists("data/traits-kinship-hannah-amelie.pkl"):
    data = pkl.load(open("data/traits-kinship-hannah-amelie.pkl", "rb"))
else:
    url = "http://ipfs.io/ipfs/QmXtFuoA3JTkhhpUpqh4dyu3FJLUG7DWAmXwUk1N2TMbAh"
    data = pkl.load(urlopen(url))

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
    

def analysis_QTL(G, pheno, kinship, pheno_name, lik, pheno_norm):
    common_samples = set(kinship.index.values).intersection(set(pheno.index.values))
    common_samples = np.asarray(list(common_samples))
    
    pheno = pheno.reindex(common_samples).copy()
    kinship = kinship.reindex(index=common_samples).copy()
    kinship = kinship.reindex(columns=common_samples).copy()
    
    assert(all(kinship.index.values == kinship.columns.values))
    assert(all(kinship.index.values == pheno.index.values))

    ok = np.isfinite(pheno.loc[:, pheno_name])
    common_samples = common_samples[ok]
    
    pheno = pheno.reindex(common_samples).copy()
    kinship = kinship.reindex(index=common_samples).copy()
    kinship = kinship.reindex(columns=common_samples).copy()
    
    assert(all(kinship.index.values == kinship.columns.values))
    assert(all(kinship.index.values == pheno.index.values))
    
    if not all(np.isfinite(pheno[pheno_name])):
        raise ValueError("not all finite: {}".format(pheno_name))
    
    y = pheno_norm(pheno.loc[:, pheno_name])
    y = pd.Series(y, index=pheno.loc[:, pheno_name].index)
    
#     phenotype_series = pd.Series(y, pheno.loc[:, pheno_name].index.values)
    obj = pkl.dumps(y)
    m = hashlib.sha1()
    m.update(obj)
    phenotype_hex = m.hexdigest()
    with open(join(dst_folder, "phenotype", phenotype_hex + ".series.pkl"), "wb") as f:
        f.write(obj)
    
    obj = pkl.dumps(kinship)
    m = hashlib.sha1()
    m.update(obj)
    kinship_hex = m.hexdigest()
    with open(join(dst_folder, "kinship", kinship_hex + ".dataframe.pkl"), "wb") as f:
        f.write(obj)

    try:
        call = "limix.qtl.scan(G, y, lik, kinship, verbose=True)"
        model = limix.qtl.scan(G, y, lik, kinship, verbose=True)
    except ValueError as e:
        print(e)
        print("Pheno name: {}".format(pheno_name))
        return None

    return dict(pv=list(asarray(model.variant_pvalues)), lik=lik, call=call, phenotype=phenotype_hex, kinship=kinship_hex,
                null_covariate_effsizes=list(asarray(model.null_covariate_effsizes)),
                variant_effsizes=list(asarray(model.variant_effsizes)), variant_effsizes_se=list(asarray(model.variant_effsizes_se)))


def analysis_null(G, pheno, kinship, pheno_name, lik, pheno_norm):
    common_samples = set(kinship.index.values).intersection(set(pheno.index.values))
    common_samples = np.asarray(list(common_samples))
    
    pheno = pheno.reindex(common_samples).copy()
    kinship = kinship.reindex(index=common_samples).copy()
    kinship = kinship.reindex(columns=common_samples).copy()
    
    assert(all(kinship.index.values == kinship.columns.values))
    assert(all(kinship.index.values == pheno.index.values))

    ok = np.isfinite(pheno.loc[:, pheno_name])
    common_samples = common_samples[ok]
    
    pheno = pheno.reindex(common_samples).copy()
    kinship = kinship.reindex(index=common_samples).copy()
    kinship = kinship.reindex(columns=common_samples).copy()
    
    assert(all(kinship.index.values == kinship.columns.values))
    assert(all(kinship.index.values == pheno.index.values))
    
    if not all(np.isfinite(pheno[pheno_name])):
        raise ValueError("not all finite: {}".format(pheno_name))
    
    y = pheno_norm(pheno.loc[:, pheno_name])
    y = pd.Series(y, index=pheno.loc[:, pheno_name].index)
    
#     phenotype_series = pd.Series(y, pheno.loc[:, pheno_name].index.values)
    obj = pkl.dumps(y)
    m = hashlib.sha1()
    m.update(obj)
    phenotype_hex = m.hexdigest()
    with open(join(dst_folder, "phenotype", phenotype_hex + ".series.pkl"), "wb") as f:
        f.write(obj)
    
    obj = pkl.dumps(kinship)
    m = hashlib.sha1()
    m.update(obj)
    kinship_hex = m.hexdigest()
    with open(join(dst_folder, "kinship", kinship_hex + ".dataframe.pkl"), "wb") as f:
        f.write(obj)
    
    random.seed(int(phenotype_hex, 16) % 100000)
    null_samples = G.coords["samples"].values.copy()
    random.shuffle(null_samples)
    G.coords["samples"] = null_samples

    R = {"pv":[]}
    for seed in range(5):
        random.seed((int(phenotype_hex, 16) + seed) % 100000)
        null_samples = G.coords["samples"].values.copy()
        random.shuffle(null_samples)
        G.coords["samples"] = null_samples

        try:
            call = "limix.qtl.scan(G, y, lik, kinship, verbose=True)"
            model = limix.qtl.scan(G, y, lik, kinship, verbose=True)
        except ValueError as e:
            print(e)
            print("Pheno name: {}".format(pheno_name))
            return None
        R["pv"] += list(asarray(model.variant_pvalues))
    
    R["call"] = call
    R["lik"] = lik

    return R


f = h5py.File("arrayexpress/HS_shuf.hdf5", "r")
samples = np.asarray(f["/imputed_genotypes/row_header/rat"].value)
samples = [i.decode() for i in samples]
f.close()

data0 = deepcopy(data)
remove = []
for name in data0['measures'].columns.values:
    if isnumber(data0['measures'].loc[:, name]) and isdiscrete(data0['measures'].loc[:, name]):
        data0['measures'].loc[:, name] = get_poisson(data0['measures'].loc[:, name])
    else:
        remove.append(name)
for name in remove:
    del data0['measures'][name]
    
    
Gs = []
for i in tqdm(range(1, 22)):
    G = xr.open_dataarray("arrayexpress/HS_shuf.hdf5", "/imputed_genotypes/chr{}".format(i))
    f = h5py.File("arrayexpress/HS_shuf.hdf5", "r")
    G = G.chunk(f["/imputed_genotypes/chr{}/array".format(i)].chunks)
    f.close()
    G = G.rename({G.dims[0]: "snps", G.dims[1]: "samples"})
    Gs.append(G)


G = xr.concat(Gs, dim="snps").T[:, np.load("maf.npy") >= MAF]
G['samples'] = samples
G['snps'] = range(G.shape[1])
        

njobs = int(sys.argv[1])
num = int(sys.argv[2])
what = sys.argv[3]

if what == "scan":
    for ii, name in enumerate(data0['measures'].columns.values):
        if ii % njobs != num:
            continue
        print("Processing: {}".format(name))
        dst_file = join(dst_folder, "scan_measures_normal_" + name + "_quantile_gaussianize.json")
        if os.path.exists(dst_file) or os.path.exists(dst_file + ".failed"):
            continue
        r = analysis_QTL(G, data0['measures'], data0['kinship'], name, 'normal', limix.qc.quantile_gaussianize)
        if r is None:
            open(dst_file + ".failed", "w").write("")
        else:
            json.dump(r, open(dst_file, "w"))
elif what == "null":
    for ii, name in enumerate(data0['measures'].columns.values):
        if ii % njobs != num:
            continue
        print("Processing: {}".format(name))
        dst_file = join(dst_folder, "null_measures_normal_" + name + "_quantile_gaussianize.json")
        if os.path.exists(dst_file) or os.path.exists(dst_file + ".failed"):
            continue
        r = analysis_null(G, data0['measures'], data0['kinship'], name, 'normal', limix.qc.quantile_gaussianize)
        if r is None:
            open(dst_file + ".failed", "w").write("")
        else:
            json.dump(r, open(dst_file, "w"))
