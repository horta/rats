import requests
import h5py
import gzip
from tqdm import tqdm
import numpy as np
import pickle as pkl
from joblib import Parallel, delayed
import sys
import os
import pdb

if len(sys.argv) == 4:

    chrid = int(sys.argv[1])
    njobs_for_chrom = int(sys.argv[2])
    jobid_for_chrom = int(sys.argv[3])


    def fetch_pos_info(chrid, pos):
        r = requests.get(f'https://rest.rgd.mcw.edu/rgdws/genes/{chrid}/{pos}/{pos}/360')
        r.raise_for_status()
        return r.json()

    with h5py.File("arrayexpress/HS_shuf.hdf5", "r") as f:
        pos = f[f"/imputed_genotypes/chr{chrid}/col_header/pos"]

        nsnps = pos.shape[0]
        chunk_size = nsnps // 100
        chunks = [chunk_size] * 100
        rem = nsnps - chunk_size * 100
        if rem > 0:
            chunks.append(rem)

        start = 0
        for (i, chunk) in tqdm(enumerate(chunks)):
            end = start + chunk

            if i % njobs_for_chrom == jobid_for_chrom:
                calls = (delayed(fetch_pos_info)(chrid, int(pos[ii])) for ii in range(start, end))
                annots = Parallel(n_jobs=10, prefer="threads", verbose=1)(calls)

                try:
                    os.makedirs(f'arrayexpress/chr{chrid}')
                except Exception:
                    pass

                with gzip.open(f'arrayexpress/chr{chrid}/{i}.pkl.gz', 'wb') as f0:
                    pkl.dump(annots, f0, -1)

            start = end

else:

    chrom_data = dict()
    for chromid in tqdm(range(1, 22)):
        files = os.listdir(f"arrayexpress/chr{chromid}")
        chrom_data[chromid] = {}
        for f in tqdm(files, leave=False):
            with gzip.open(f'arrayexpress/chr{chromid}/{f}', 'rb') as f0:
                chrom_data[chromid][int(f[:-7])] = pkl.load(f0)

    pos_store = {}
    with h5py.File("arrayexpress/HS_shuf.hdf5", "r") as hs_f:
        for chromid in tqdm(range(1, 22)):
            pos = np.asarray(hs_f[f"/imputed_genotypes/chr{chromid}/col_header/pos"], int)
            pos_store[chromid] = {"pos": {}, "idx": [None] * len(pos)}
            ii = 0
            chunk_idxs = sorted(chrom_data[chromid].keys())
            for chunkid in tqdm(chunk_idxs, leave=False):
                for v0 in chrom_data[chromid][chunkid]:
                    data = (pos[ii], v0)
                    pos_store[chromid]["idx"][ii] = data
                    pos_store[chromid]["pos"][pos[ii]] = data
                    ii += 1
            if ii != len(pos):
                pdb.set_trace()
                raise ValueError()

    with gzip.open(f'arrayexpress/pos.pkl.gz', 'wb') as f:
        pkl.dump(pos_store, f, -1)