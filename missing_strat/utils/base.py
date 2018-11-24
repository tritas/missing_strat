import numpy as np
import pandas as pd


def hstack_missing_indicator(serie, missing=0, indicator=1):
    # TODO: Handle existing zeros with the missing value parameter
    indicator_serie = indicate_missing(serie, missing=indicator)
    filled = serie.fillna(missing)
    indicator_stacked = pd.DataFrame([filled, indicator_serie]).T
    return indicator_stacked


def indicate_missing(serie, missing=1):
    if not isinstance(serie, pd.Series):
        return NotImplementedError(type(serie))
    indicator_arr = np.zeros_like(serie)
    indicator_arr[serie.isnull()] = missing
    indicator_serie = pd.Series(
        data=indicator_arr, name=serie.name + "_missing", index=serie.index
    )
    return indicator_serie


def impute_weighted_mean(serie, weights, inplace=True):
    r""" Impute weighted average to numerical series with missing vals.

    Notes
    -----
    Biased estimation - this is not the real weighted avg.
    \bar{x} = \sum_i w_i x_i / \sum_i w_i
    For very few missing values it's correct up to ~1e-8 """
    imp_serie = serie if inplace else serie.copy()
    mask = serie.isnull()
    imp_serie.loc[mask] = np.average(serie[~mask], weights=weights[~mask])
    return imp_serie
