# coding=utf-8
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.utils import assert_all_finite

from .. import imputation_strategies
from ..inference.independent import infer_missing
from ..utils.base import impute_weighted_mean


class MultiImputer(TransformerMixin, BaseEstimator):
    """ Does not impute values for categorical data as missingness is
    one-hot-encoded later. """

    def __init__(
        self,
        figures_dir: Optional[str] = None,
        strategy: str = "mean",
        verbose: bool = False,
        numerical: Optional[Tuple[str]] = (),
        categorical: Optional[Tuple[str]] = (),
        ordinal: Optional[Tuple[str]] = (),
        boolean: Optional[Tuple[str]] = (),
    ):
        """
        Parameters
        ----------
        figures_dir: str
        strategy: str
          The strategy to use for imputation
        verbose: bool
        """
        assert strategy in imputation_strategies
        logger.info("[Imputation] Strategy is `{}`".format(strategy))
        self.figures_dir = figures_dir
        self.strategy = strategy
        self.verbose = verbose
        self.sample_weight = None
        self.numerical_to_impute = numerical + ordinal
        self.categorical_to_impute = categorical
        self.boolean_cols = boolean

    def _impute(self, df: pd.DataFrame, column: str, estimator: str) -> pd.Series:
        """Estimate missing values with classification or regression"""
        assert estimator in ("clf", "reg"), estimator
        logger.debug(
            "[Imputation] On column {} with `{}` estimator".format(column, estimator)
        )

        if data.loc[:, column].dtype.type in (np.str, np.object_):
            labels, uniques = data.loc[:, column].factorize()
            labels = labels.astype(np.float_)
            labels[labels == -1] = np.nan
            data.loc[:, column] = labels
            logger.debug(uniques)
        else:
            uniques = None

        imputed_series = infer_missing(data, column, estimator, self.figures_dir)

        if uniques is not None:
            imputed_series.loc[:] = uniques[imputed_series.values.astype(np.int_)]
        return imputed_series

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sample_weight: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **fit_params
    ) -> "MultiImputer":
        self.sample_weight = sample_weight
        return self

    def transform(
        self,
        df: pd.DataFrame,
        sample_weight: Optional[
            Union[str, List, pd.Series, np.ndarray]
        ] = "sample_weight",
    ) -> pd.DataFrame:
        incomplete_columns = df.columns[df.isnull().sum(axis=0) != 0].values
        missing_categorical = np.intersect1d(
            incomplete_columns, self.categorical_to_impute
        )
        missing_numerical_cols = np.intersect1d(
            incomplete_columns, self.numerical_to_impute
        )
        missing_boolean = np.intersect1d(incomplete_columns, self.boolean_cols)
        logger.debug(
            "[Imputation Pipeline] Missing categorical: \n{}".format(
                missing_categorical
            )
        )
        logger.debug(
            "[Imputation Pipeline] Missing numerical: \n{}".format(
                missing_numerical_cols
            )
        )
        # Retrieve weights if passed in DataFrame and not
        if sample_weight is not None:
            if isinstance(sample_weight, str) and sample_weight in df.columns:
                self.sample_weight = df.loc[:, weight_column]
            elif isinstance(sample_weight, (np.ndarray, pd.Series)):
                self.sample_weight = sample_weight
            else:
                # Try to convert to np.array
                self.sample_weight = np.asarray(sample_weight)
            assert len(self.sample_weight) == len(
                df
            ), "Dimensions {} and {} don't match".format(
                len(self.sample_weight), len(df)
            )

        if self.strategy == "fill":
            df.fillna(0, inplace=True)
        elif self.strategy == "mean" and self.sample_weight is not None:
            #  Impute mean such that resulting weighted average is zero
            assert np.all(
                df.loc[:, missing_numerical_cols].dtypes == np.float_
            ), "\n" + str(df.loc[:, missing_numerical_cols].head())
            df.loc[:, missing_numerical_cols] = df.loc[:, missing_numerical_cols].apply(
                lambda s: impute_weighted_mean(s, self.sample_weight)
            )
        elif self.strategy in ("mean", "median", "most_frequent"):
            # Impute numerical data only
            # TODO: Handle Categoricals
            # TODO: Handle Booleans
            imp = Imputer(strategy=self.strategy, verbose=self.verbose, copy=False)
            try:
                df.loc[:, missing_numerical_cols] = imp.fit_transform(
                    df.loc[:, missing_numerical_cols].values
                )
            except BaseException:
                logger.warning("\n" + str(df.loc[:, missing_numerical_cols].head()))
                raise
        elif self.strategy == "infer":
            imputation_dict = dict()
            # Predict missing values
            for clf_column in missing_categorical + missing_boolean:
                imputation_dict[clf_column] = self._impute(df, clf_column, "clf")

            for reg_column in missing_numerical_cols:
                imputation_dict[reg_column] = self._impute(df, reg_column, "reg")

            # Actually replace values
            for column, serie in imputation_dict.items():
                if serie is not None:
                    df.loc[:, column] = serie.values
        else:
            return NotImplementedError(self.strategy)

        logger.debug(
            "[Imputation pipeline] Final columns:\n{}".format(
                sorted(df.columns.tolist())
            )
        )
        assert_all_finite(df)

        return df
