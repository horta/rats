import pandas as pd
from os.path import join
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

folder = "/ipfs/QmbBGQnqoEjxBuoq5eruMdExNxbruYkKMkNR9NwDuyKdGT/processeddata/"
ipheno = join(folder, "imputed_phenotypes.rds")
readRDS = robjects.r["readRDS"]
df = readRDS(ipheno)
# $imp$phenix$imp
m = pandas2ri.ri2py(df[0][0][0])
df0 = pd.DataFrame(
    data=m, columns=list(df[0][0][0].colnames), index=list(df[0][0][0].rownames)
)
# print(df.head())
