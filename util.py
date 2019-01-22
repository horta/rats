import pandas as pd

def get_amelie_selection():
    return list(pd.read_csv("amelie trait selection.txt", header=None).values.ravel())