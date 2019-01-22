import pdb
import h5py
import numpy as np
import dask.array as da
from tqdm import tqdm
import limix

mafs = []
with h5py.File("arrayexpress/HS_shuf.hdf5", "r") as f:
    for i in tqdm(range(1, 22)):
        G = f[f"/imputed_genotypes/chr{i}/array"]
        G = da.from_array(G, chunks=G.chunks)
        mafs.append(np.asarray(limix.qc.compute_maf(G.T)))

mafs = np.concatenate(mafs)
np.save("maf", mafs)