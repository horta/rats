import pdb
import numpy as np
from tqdm import tqdm
import pickle as pkl
import gzip
from numpy import log10
from bokeh.models import ColumnDataSource, HoverTool, CDSView, IndexFilter
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.iris import flowers
import pandas as pd
from numpy import unique, flipud, cumsum


def _isint(i):
    try:
        int(i)
    except ValueError:
        return False
    else:
        return True


def _chr_precedence(df):
    uchr = unique(df["chr"].values)
    nchr = [int(i) for i in uchr if _isint(i)]
    if len(nchr) > 0:
        offset = max(nchr)
    else:
        offset = -1
    precedence = {str(i): i for i in nchr}
    schr = sorted([i for i in uchr if not _isint(i)])
    for i, s in enumerate(schr):
        precedence[s] = offset + i + 1
    return precedence


def _abs_pos(df):
    order = df["order"].unique()
    chrom_ends = [df["pos"][df["order"] == c].max() for c in order]

    offset = flipud(cumsum(chrom_ends)[:-1])

    df["abs_pos"] = df["pos"].copy()

    order = list(reversed(order))
    for i, oi in enumerate(offset):
        ix = df["order"] == order[i]
        df.loc[ix, "abs_pos"] = df.loc[ix, "abs_pos"] + oi

    return df


def _chrom_bounds(df):
    order = df["order"].unique()
    v = []
    for c in order:
        vals = df["abs_pos"][df["order"] == c]
        v += [(vals.min(), vals.max())]
    return v


def bokeh_manhattan(df, out_fp):
    # print("Reading pv...")
    # df = pd.read_pickle("df_pv.pkl")
    # print("Done")

    print("Reading pos...")
    with gzip.open("arrayexpress/pos.pkl.gz", "rb") as f:
        pos = pkl.load(f)
    print("Done")

    df["chr"] = df["chr"].astype(str)
    df["pos"] = df["pos"].astype(int)
    df["pv"] = df["pv"].astype(float)
    df = df[df["pv"] <= 0.01].copy()

    chr_order = _chr_precedence(df)
    df = df.assign(order=[chr_order[i] for i in df["chr"].values])
    df = df.sort_values(by=["order", "pos"])


    def _get(v, name):
        if len(v) == 0:
            return "unknown"
        if len(v) == 1:
            return str(v[0][name])
        return "; ".join(str(vi[name]) for vi in v)


    keys = [
        "name",
#        "description",
#        "product",
#        "function",
#        "notes",
#        "rgdId",
#        "type",
#        "nomenReviewDate",
#        "speciesTypeKey",
#        "refSeqStatus",
#        "soAccId",
#        "ncbiAnnotStatus",
#        "variant",
    ]

    df = _abs_pos(df)
    df["-logpv"] = -log10(df["pv"])
    data = {"-logpv": df["-logpv"], "abs_pos": df["abs_pos"], "pos": df["pos"]}
    data["chr"] = df["chr"]

    for k in tqdm(keys):
        data[k] = []
        for chromid in range(1, 22):
            P = pos[int(chromid)]["pos"]
            D = df.query(f"chr == '{chromid}'")["pos"]
            data[k] += [_get(P[p][1], k) for p in D]

    ds = ColumnDataSource(data=data)

    tools = ["box_zoom", "reset"]
    p = figure(title="pv", tools=tools,
               plot_width=1200, plot_height=400)
    tooltips = [("-logpv", "@{-logpv}"),
                ("chr", "@chr"), ("pos", "@pos")]
    for k in keys:
        tooltips.append((k, f"@{k}"))
    p.add_tools(HoverTool(tooltips=tooltips))
    p.xaxis.axis_label = 'chrom'
    p.yaxis.axis_label = '-log10(pv)'
    colors = ["red", "blue"]

    for i in range(2):
        filters = []
        for chromid in range(1, 22):
            if chromid % 2 == i:
                ii = np.where(df["chr"].astype(int) == chromid)[0]
                filters += list(ii)
        idx = IndexFilter(filters)
        view = CDSView(source=ds, filters=[idx])
        p.circle(source=ds, x="abs_pos", y="-logpv", fill_alpha=0.2, size=3, view=view, color=colors[i])


    output_file(out_fp, title="pv")

    show(p)
