import numpy as np
import polars as pl
import rpy2.rinterface_lib
from rpy2.robjects import r as R


def load_df_from_R(code):
    df = R(code)
    if isinstance(df.names, rpy2.rinterface_lib.sexp.NULLType):
        return pl.DataFrame(np.array(df))
    return pl.DataFrame(np.array(df), index=df.names[0], columns=df.names[1])
