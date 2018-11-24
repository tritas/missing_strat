import logging
from os.path import join

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def infer_missing(df, target_column, inference_type, figures_dir, verbose=False):
    """Imputed infered values for a columns with missing values
    by gradient boosted trees regression or classification

    Parameters
    ----------
    df : pandas dataframe

    target_column: string
        The column to impute values for.

    inference_type: string
        The type of inference: 'reg' for regression, 'clf' for classification

    figures_dir: filepath
         File path to the directory where feature importance figures
         will be stored.
    verbose: bool

    Returns
    -------
    df: pandas dataframe
        The input dataframe completed with infered values.
    """
    # TODO: Hyperopt the CV'ed version of this function
    if inference_type not in ("reg", "clf"):
        raise ValueError(inference_type)

    # Remove some variables having the same prefix with target
    # to prevent leaking data from added & related vars
    target_prefix = target_column[:3]
    input_columns = [c for c in df.columns if not c.startswith(target_prefix)]
    # Make X, y
    missing_mask = pd.isnull(df.loc[:, target_column])
    y_full = df.loc[~missing_mask, target_column]
    # One-hot encode string columns
    X = pd.get_dummies(df.loc[:, input_columns], dummy_na=True)
    X_missing = X.loc[missing_mask, :]
    X_full = X.loc[~missing_mask, :]

    ax, fig = plt.subplots(1, 1, figsize=rect_figsize)
    y_full.hist(
        bins="auto", normed=True, alpha=0.4, color="grey", label="Original values"
    )
    # Make train/test split
    if inference_type == "clf":
        # Some classes are rare, here we artificially change the labels
        # to the nearest neighbourghs
        labels, class_counts = np.unique(y_full, return_counts=True)
        for i, (label, count) in enumerate(zip(labels, class_counts)):
            if count < 2:
                y_full[y_full == label] = labels[i - 1]
        stratify = y_full
    else:
        try:
            # Stratify by quantiles if possible
            stratify, _ = pd.factorize(pd.qcut(y_full, 20, duplicates="drop"))
        except ValueError:
            stratify = None
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_full, y_full, test_size=0.5, random_state=seed, stratify=stratify
        )
    except ValueError:
        logger.warning(
            "[Imputation] Stratified split failed for {}".format(target_column)
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_full, y_full, test_size=0.5, random_state=seed, stratify=None
        )

    logger.info(
        "[Imputation] Column {}, n_missing={}/{}, train/test={}/{}".format(
            target_column, missing_mask.sum(), len(X), len(X_train), len(X_valid)
        )
    )
    # Choose model
    if inference_type == "clf":
        booster = xgb.XGBClassifier(seed=seed)
    else:
        booster = xgb.XGBRegressor(seed=seed)

    # booster = xgb.cv(param, dtrain, num_round, nfold=20, stratified=True,
    #                  metrics=['error'], seed=seed,
    #                  callbacks=[xgb.callback.print_evaluation(show_stdv=True),
    #                             xgb.callback.early_stop(3)])

    # Fit, predict
    booster.fit(
        X_train,
        y_train,
        early_stopping_rounds=1,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )
    # Write back model prediction
    preds = booster.predict(X_missing, ntree_limit=booster.best_iteration)
    imputed_serie = df.loc[:, target_column].copy()
    imputed_serie.loc[missing_mask] = preds

    pd.Series(preds).hist(
        bins="auto", normed=True, alpha=0.4, color="red", label="Predictions"
    )
    imputed_serie.hist(
        bins="auto", normed=True, alpha=0.4, color="blue", label="Completed values"
    )
    plt.title("Infered missing values for column `{}`".format(target_column))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(join(figures_dir, "infered_values_histo_{}.png".format(target_column)))
    plt.close()

    metrics = booster.evals_result()
    fig, ax = plt.subplots(1, 1, figsize=rect_figsize)
    train_metrics = metrics["validation_0"]
    for k, v in train_metrics.items():
        plt.plot(np.arange(len(v)), v, label="train " + k, color="grey")
    train_metrics = metrics["validation_1"]
    for k, v in train_metrics.items():
        plt.plot(np.arange(len(v)), v, label="test " + k, color="red")
    plt.legend()
    plt.title("Estimator performance evolution for column `{}`".format(target_column))
    plt.xlabel("No.Iterations")
    plt.ylabel("Error value (a.u.)")
    plt.savefig(
        join(figures_dir, "metrics_{}_{}.png".format(target_column, inference_type))
    )
    plt.close()

    for weight in ("weight", "gain", "cover"):
        fig, ax = plt.subplots(1, 1, figsize=big_square)
        xgb.plot_importance(booster, ax, importance_type=weight)
        figure_fn = "feature_importance_{}_{}.png".format(weight, target_column)
        figure_path = join(figures_dir, figure_fn)
        plt.xlabel(weight.capitalize() + " value")
        plt.ylabel("Attribute")
        plt.title("Var. importance for predicting {}".format(target_column))
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.clf()
        plt.close()

    return imputed_serie
