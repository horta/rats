from tqdm import tqdm
import pickle as pkl
from glob import glob
import os
import sys
import simplejson as json

folder = sys.argv[1]
force = False
delete_after = False

if len(sys.argv) > 2:
    if "--force" in set(sys.argv[2:]):
        force = True
    if "--delete-after" in set(sys.argv[2:]):
        delete_after = True

for f in tqdm(glob(os.path.join(folder, "*.json"))):
    if not force and os.path.exists(f + ".pkl"):
        continue
    data = json.load(open(f, "rb"))
    pkl.dump(data, open(f + ".pkl", "wb"), -1)
    if delete_after:
        os.remove(f)
