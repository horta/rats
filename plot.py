import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from poisson import poisson_similarity, is_discrete
# from util import get_amelie_selection

def get_amelie_selection():
    return list(pd.read_csv("amelie trait selection.txt", header=None).values.ravel())

def plot_poisson_ordered(df):
    ys = []
    sim = []
    for (i, v) in enumerate(df.columns.values):
        try:
            np.asarray(df[v].values, float)
            numeric = True
        except ValueError:
            numeric = False
        
        if numeric:
            vals = df[v].astype(float).dropna()
        else:
            vals = df[v]
        ys.append(vals.copy())
        sim.append(poisson_similarity(vals))
    
    ys = [v[1] for v in sorted(enumerate(ys), key=lambda x: sim[x[0]])]
    sims = list(reversed(sorted(sim)))
    ys = list(reversed(ys))
    amelie_selection = get_amelie_selection()
    plt.figure(figsize=(20, 250))
    for (i, y) in enumerate(ys):
        plt.subplot(54, 4, i + 1)
        try:
            try:
                np.asarray(y, float)
                numeric = True
            except ValueError:
                numeric = False
            if numeric:
                sns.distplot(y.astype(float).dropna())
            else:
                plt.xlabel(y.name)
        except (np.linalg.LinAlgError, TypeError) as e:
            print("{}: {}".format(type(e).__name__, y.name))
            plt.gca().set_facecolor((255/255., 217/255., 209/255.))
            plt.xlabel(y.name)
        else:
            if sims[i] == 0.0:
                plt.gca().set_facecolor((0.95, 0.95, 0.95))
        if y.name in amelie_selection:
            plt.gca().xaxis.label.set_color('red')
        
        if numeric:
            if not is_discrete(y.values):
                plt.text(0.5, 0.5, "CONTINUOUS",
                        transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "NON-NUMERIC",
                        transform=plt.gca().transAxes)
    plt.savefig("fig/trait-distplots.pdf")

def plot_distplots(df, height):

    ys = []
    for (i, v) in enumerate(df.columns.values):
        try:
            np.asarray(df[v].values, float)
            numeric = True
        except ValueError:
            numeric = False
        
        if numeric:
            vals = df[v].astype(float).dropna()
        else:
            vals = df[v]
        ys.append(vals.copy())

    amelie_selection = get_amelie_selection()
    plt.figure(figsize=(20, height))
    ncols = df.shape[1]
    nrows = ncols//4 + ((ncols%4) != 0)
    for (i, y) in enumerate(ys):
        plt.subplot(nrows, 4, i + 1)
        try:
            try:
                np.asarray(y, float)
                numeric = True
            except ValueError:
                numeric = False
            if numeric:
                sns.distplot(y.astype(float).dropna())
            else:
                plt.xlabel(y.name)
        except (np.linalg.LinAlgError, TypeError) as e:
            print("{}: {}".format(type(e).__name__, y.name))
            plt.gca().set_facecolor((255/255., 217/255., 209/255.))
            plt.xlabel(y.name)
        if y.name in amelie_selection:
            plt.gca().xaxis.label.set_color('red')
        
        if numeric:
            if not is_discrete(y.values):
                plt.text(0.5, 0.5, "CONTINUOUS",
                        transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "NON-NUMERIC",
                        transform=plt.gca().transAxes)
    plt.show()
    
def plot_distplot(series, ax):

    y = series.astype(float).dropna().values
    sns.distplot(y)
    ax.set_xlabel(series.name)
    
    amelie_selection = get_amelie_selection()
    if series.name in amelie_selection:
        ax.xaxis.label.set_color('red')
