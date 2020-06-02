#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:29:34 2019

@author: yinzhuo
"""

import os
import time
import pickle
import warnings
from datetime import datetime
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from numpy import iinfo, finfo, int8, int16, int32, int64, float32, float64
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    
    Refs:
        [1] https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        [2] https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km * 1000


def dynamic_time_warping(X, Y):
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------

    @Description:
    ----------
    Dynamic time warping distance for trajectory similarity calculation.

    @Parameters:
    ----------
    X: list-like or numpy array-like
        trajectory X.
    Y: list-like or numpy array-like
        trajectory Y.

    @Return:
    ----------
    dtw distance between series X and Y.
    """
    length_X, length_Y = len(X), len(Y)

    # Initializing some parameters
    dp = np.zeros((length_X + 1, length_Y + 1))
    dp[0, 1:] = np.inf
    dp[1:, 0] = np.inf
    dp_panel = dp[1:, 1:]

    # Initializing the distance matrix
    for i in range(length_X):
        for j in range(length_Y):
            dp_panel[i, j] = np.linalg.norm(X[i, :] - Y[j, :])

    # Calculation of the dp matrix
    for i in range(length_X):
        for j in range(length_Y):
            dp_panel[i, j] += min(dp[i+1, j], dp[i, j+1], dp[i, j])
    return dp[-1, -1]


def timefn(fcn):
    """Decorator for efficency analysis. """
    @wraps(fcn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fcn.__name__ + " took {:.5f}".format(end-start)
            + " seconds.")
        return result
    return measure_time


@timefn
def basic_feature_report(data_table=None, precent=None):
    """Reporting basic characteristics of the tabular data data_table."""
    precent = precent or [0.01, 0.25, 0.5, 0.75, 0.95, 0.9995]
    if data_table is None:
        return None
    num_samples = len(data_table)

    # Basic statistics
    basic_report = data_table.isnull().sum()
    basic_report = pd.DataFrame(basic_report, columns=["#missing"])
    basic_report["missing_precent"] = basic_report["#missing"]/num_samples
    basic_report["#uniques"] = data_table.nunique(dropna=False).values
    basic_report["types"] = data_table.dtypes.values
    basic_report.reset_index(inplace=True)
    basic_report.rename(columns={"index":"feature_name"}, inplace=True)

    # Basic quantile of data
    data_description = data_table.describe(precent).transpose()
    data_description.reset_index(inplace=True)
    data_description.rename(columns={"index":"feature_name"}, inplace=True)
    basic_report = pd.merge(basic_report, data_description,
        on='feature_name', how='left')
    return basic_report


class LoadSave():
    """Class for loading and saving object in .pkl format."""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def save_data(self, data=None, path=None):
        """Save data to path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        self.__save_data(data)

    def load_data(self, path=None):
        """Load data from path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        return self.__load_data()

    def __save_data(self, data=None):
        """Save data to path."""
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._file_name))
        with open(self._file_name, 'wb') as file:
            pickle.dump(data, file)
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")

    def __load_data(self):
        """Load data from path."""
        if not self._file_name:
            raise ValueError("Invaild file path !")
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._file_name))
        with open(self._file_name, 'rb') as file:
            data = pickle.load(file)
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data


class ReduceMemoryUsage():
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Reduce the memory usage of pandas dataframe.
    
    @Parameters:
    ----------
    data: pandas DataFrame-like
        The dataframe that need to be reduced memory usage.
    verbose: bool
        Whether to print the memory reduction information or not.
        
    @Return:
    ----------
    Memory-reduced dataframe.
    """
    def __init__(self, data_table=None, verbose=True):
        self._data_table = data_table
        self._verbose = verbose

    def type_report(self, data_table):
        """Reporting basic characteristics of the tabular data data_table."""
        data_types = list(map(str, data_table.dtypes.values))
        basic_report = pd.DataFrame(data_types, columns=["types"])
        basic_report["feature_name"] = list(data_table.columns)
        return basic_report

    @timefn
    def reduce_memory_usage(self):
        memory_reduced_data = self.__reduce_memory()
        return memory_reduced_data

    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memory_before_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        types = self.type_report(self._data_table)
        if self._verbose is True:
            print("@Memory usage of data is {:.5f} MB.".format(
                memory_before_reduced))

        # Scan each feature in data_table, reduce the memory usage for features
        for ind, name in enumerate(types["feature_name"].values):
            # ToBeFixed: Unstable query.
            feature_type = str(
                types[types["feature_name"] == name]["types"].iloc[0])

            if (feature_type in "object") and (feature_type in "datetime64[ns]"):
                try:
                    feature_min = self._data_table[name].min()
                    feature_max = self._data_table[name].max()

                    # np.iinfo for reference:
                    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                    # numpy data types reference:
                    # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                    if "int" in feature_type:
                        if feature_min > iinfo(int8).min and feature_max < iinfo(int8).max:
                            self._data_table[name] = self._data_table[name].astype(int8)
                        elif feature_min > iinfo(int16).min and feature_max < iinfo(int16).max:
                            self._data_table[name] = self._data_table[name].astype(int16)
                        elif feature_min > iinfo(int32).min and feature_max < iinfo(int32).max:
                            self._data_table[name] = self._data_table[name].astype(int32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(int64)
                    else:
                        if feature_min > finfo(float32).min and feature_max < finfo(float32).max:
                            self._data_table[name] = self._data_table[name].astype(float32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(float64)
                except Exception as error_msg:
                    print("\n--------ERROR INFORMATION---------")
                    print(error_msg)
                    print("Error on the {}".format(name))
                    print("--------ERROR INFORMATION---------\n")
            if self._verbose is True:
                print("Processed {} feature({}), total is {}.".format(
                    ind + 1, name, len(types)))

        memory_after_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        if self._verbose is True:
            print("@Memory usage after optimization: {:.5f} MB.".format(
                memory_after_reduced))
            print("@Decreased by {:.5f}%.".format(
                100 * (memory_before_reduced - memory_after_reduced) / memory_before_reduced))
        print("-------------------------------------------")
        return self._data_table


def training_res_to_log(training_res=None, comment=None):
    PATH = ".//submission_logs//"
    log_names = os.listdir(PATH)

    if "log.csv" not in log_names:
        df = pd.DataFrame(None, columns=["date_time", "train_id", "folds",
                                         "valid_f1", "valid_acc", "best_iters",
                                         "comments"])
    else:
        df = pd.read_csv(PATH + "log.csv")
    curr_res = {}
    curr_res["date_time"] = datetime.now().__str__()[:-4]
    curr_res["train_id"] = len(df)
    curr_res["folds"] = training_res["fold"].max() + 1
    curr_res["valid_f1"] = training_res["valid_f1"].mean()
    curr_res["valid_acc"] = training_res["valid_acc"].mean()

    curr_res["best_iters"] = training_res["best_iters"].mean()
    curr_res["comments"] = comment

    df.loc[len(df)] = curr_res
    df.to_csv(PATH + "log.csv", index=False)
    return df.iloc[::-1]


def lgb_clf_training(train, test, params=None, num_folds=3,
                     stratified=False, shuffle=True, random_state=113,
                     early_stop_rounds=100, mode="cv", mode_iters=250):
    """LightGBM quick training."""
    # Specify the cross validated method
    if stratified == True:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=shuffle,
                                random_state=random_state)
    else:
        folds = KFold(n_splits=num_folds, shuffle=shuffle,
                      random_state=random_state)

    # Make oof predictions, test predictions
    importances = pd.DataFrame(None)
    importances["feature_name"] = list(train.drop(["boat_id",
        "target"], axis=1).columns)
    score = np.zeros((num_folds, 8))

    oof_pred = np.zeros((len(train), 3))
    y_pred = np.zeros((len(test), 3))

    # Start training
    train_data, train_label = train.drop(["boat_id", "target"], axis=1), train["target"]
    ###########################################################################
    for fold, (train_id, val_id) in enumerate(folds.split(train_data, train_label)):
        # iloc the split data
        X_train, y_train = train_data.loc[train_id], train_label.loc[train_id]
        X_valid, y_valid = train_data.loc[val_id], train_label.loc[val_id]
        params["random_state"] = random_state + fold * 100

        # Start training
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values,
                                                           y_train.values),
                                                          (X_valid.values,
                                                           y_valid.values)],
                                                          early_stopping_rounds=early_stop_rounds,
                                                          verbose=False)
        importances["lgb_importances_" + str(fold)] = clf.feature_importances_
        train_pred_proba = clf.predict_proba(X_train.values,
            num_iteration=clf.best_iteration_-1)
        valid_pred_proba = clf.predict_proba(X_valid.values,
            num_iteration=clf.best_iteration_-1)

        score_tmp = clf.evals_result_
        score_keys = list(score_tmp.keys())
        score[fold, 0] = fold
        score[fold, 1] = f1_score(y_train.values, np.argmax(train_pred_proba,
             axis=1), average="macro")
        score[fold, 2] = f1_score(y_valid.values, np.argmax(valid_pred_proba,
             axis=1), average="macro")
        score[fold, 3] = accuracy_score(y_train.values,
            np.argmax(train_pred_proba, axis=1))
        score[fold, 4] = accuracy_score(y_valid.values,
            np.argmax(valid_pred_proba, axis=1))
        score[fold, 5] = score_tmp[
            score_keys[0]]["multi_logloss"][clf.best_iteration_-1]
        score[fold, 6] = score_tmp[
            score_keys[1]]["multi_logloss"][clf.best_iteration_-1]
        score[fold, 7] = clf.best_iteration_ - 1
        print("-- folds {}, valid f1 {:.5f}, valid acc {:.5f}.".format(fold+1,
             score[fold, 2], score[fold, 4]))
        # Get the oof_prediction and the test prediction results
        oof_pred[val_id, :] = valid_pred_proba
        y_pred += clf.predict_proba(test.drop(["boat_id"], axis=1).values,
            num_iteration=clf.best_iteration_-1) / num_folds

    score = pd.DataFrame(score, columns=["fold", "train_f1", "valid_f1",
                                         "train_acc", "valid_acc",
                                         "train_logloss", "valid_logloss",
                                         "best_iters"])
    print("\n@Average cross validation score:")
    print("=================================")
    print("-- CV f1 std: {:.5f}, acc std: {:.5f}".format(score["valid_f1"].std(),
          score["valid_acc"].std()))
    print("-- CV logloss: {:.5f}".format(
        log_loss(train_label.values, oof_pred)))
    print("-- CV Accuracy score: {:.5f}".format(
        accuracy_score(train_label.values, np.argmax(oof_pred, axis=1))))
    print("-- CV F1 score: {:.5f}".format(
        f1_score(train_label.values, np.argmax(oof_pred, axis=1), average="macro")))
    print("=================================")

    if mode == "all":
        print("\n@LightGBM FULL-TRAINING started:")
        print("-----------------------------")
        params["n_estimators"] = mode_iters
        clf = lgb.LGBMClassifier(**params)
        clf.fit(train_data.values, train_label.values,
                eval_set=(train_data.values, train_label.values), verbose=False)
        y_pred = clf.predict_proba(test.drop(["boat_id"], axis=1).values)
        print("-----------------------------")

    # Constructing results
    oof_pred = pd.DataFrame(oof_pred, columns=["pred_" + str(i) for i in range(3)])
    y_pred = pd.DataFrame(y_pred, columns=["pred_" + str(i) for i in range(3)])
    oof_pred["boat_id"], y_pred["boat_id"] = train["boat_id"], test["boat_id"]
    return score, importances, oof_pred, y_pred


def xgb_clf_training(train, test, params=None, num_folds=3,
                     stratified=False, shuffle=True, random_state=113,
                     early_stop_rounds=100, mode="cv", mode_iters=270):
    """Xgboost quick training."""
    if stratified == True:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=shuffle,
                                random_state=random_state)
    else:
        folds = KFold(n_splits=num_folds, shuffle=shuffle,
                      random_state=random_state)

    # Make oof predictions, test predictions
    importances = pd.DataFrame(None)
    importances["feature_name"] = list(train.drop(["boat_id",
        "target"], axis=1).columns)
    score = np.zeros((num_folds, 8))

    oof_pred = np.zeros((len(train), 3))
    y_pred = np.zeros((len(test), 3))

    # Start training
    train_data, train_label = train.drop(["boat_id", "target"], axis=1), train["target"]
    ###########################################################################
    ###########################################################################
    for fold, (train_id, val_id) in enumerate(folds.split(train_data, train_label)):
        # iloc the split data
        X_train, y_train = train_data.loc[train_id], train_label.loc[train_id]
        X_valid, y_valid = train_data.loc[val_id], train_label.loc[val_id]
        params["random_state"] = random_state + fold * 50

        # Start training
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train.values, y_train.values, eval_set=[(X_train.values, y_train.values),
                                                          (X_valid.values, y_valid.values)],
                                                          early_stopping_rounds=early_stop_rounds,
                                                          verbose=False)
    
        importances["xgb_importances_" + str(fold)] = clf.feature_importances_
        train_pred_proba = clf.predict_proba(X_train.values,
            ntree_limit=clf.best_iteration - 1)
        valid_pred_proba = clf.predict_proba(X_valid.values,
            ntree_limit=clf.best_iteration - 1)

        score_tmp = clf.evals_result()
        score_keys = list(score_tmp.keys())
        score[fold, 0] = fold
        score[fold, 1] = f1_score(y_train.values, np.argmax(train_pred_proba,
             axis=1), average="macro")
        score[fold, 2] = f1_score(y_valid.values, np.argmax(valid_pred_proba,
             axis=1), average="macro")
        score[fold, 3] = accuracy_score(y_train.values,
            np.argmax(train_pred_proba, axis=1))
        score[fold, 4] = accuracy_score(y_valid.values,
            np.argmax(valid_pred_proba, axis=1))
        score[fold, 5] = score_tmp[
            score_keys[0]]["merror"][clf.best_iteration-1]
        score[fold, 6] = score_tmp[
            score_keys[1]]["merror"][clf.best_iteration-1]
        score[fold, 7] = clf.best_iteration - 1
        print("-- folds {}, valid f1 {:.5f}, valid acc {:.5f}.".format(fold+1,
             score[fold, 2], score[fold, 4]))

        # Get the oof_prediction and the test prediction results
        oof_pred[val_id, :] = valid_pred_proba
        y_pred += clf.predict_proba(test.drop(["boat_id"], axis=1).values,
            ntree_limit=clf.best_iteration-1) / num_folds

    if mode == "all":
        print("\n@XGBoost FULL-TRAINING started:")
        print("-----------------------------")
        params["n_estimators"] = mode_iters
        clf = xgb.XGBClassifier(**params)
        clf.fit(train_data.values, train_label.values,
                eval_set=[(train_data.values, train_label.values)], verbose=False)
        y_pred = clf.predict_proba(test.drop(["boat_id"], axis=1).values)
        print("-----------------------------")

    score = pd.DataFrame(score, columns=["fold", "train_f1", "valid_f1",
                                         "train_acc", "valid_acc",
                                         "train_logloss", "valid_logloss",
                                         "best_iters"])
    print("\n@Average cross validation score:")
    print("=================================")
    print("-- cross valid f1 std: {:.5f}, acc std: {:.5f}".format(score["valid_f1"].std(),
          score["valid_acc"].std()))
    print("-- CV logloss: {:.5f}".format(
        log_loss(train_label.values, oof_pred)))
    print("-- CV Accuracy score: {:.5f}".format(
        accuracy_score(train_label.values, np.argmax(oof_pred, axis=1))))
    print("-- CV F1 score: {:.5f}".format(
        f1_score(train_label.values, np.argmax(oof_pred, axis=1),
                 average="macro")))
    print("=================================")

    # Constructing results
    oof_pred = pd.DataFrame(oof_pred, columns=["pred_" + str(i) for i in range(3)])
    y_pred = pd.DataFrame(y_pred, columns=["pred_" + str(i) for i in range(3)])
    oof_pred["boat_id"], y_pred["boat_id"] = train["boat_id"], test["boat_id"]
    return score, importances, oof_pred, y_pred
