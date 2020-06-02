#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:38:54 2020

@author: yinzhuo
"""

import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from utils import LoadSave
from utils import lgb_clf_training, xgb_clf_training, training_res_to_log
import multiprocessing as mp

np.random.seed(57920)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def load_data(name=None):
    """Loading *.pkl data from .//tcdata_tmp//"""
    assert name is not None, "Invalid file name!"
    file_processor = LoadSave()
    return file_processor.load_data(path=".//tcdata_tmp//{}".format(name))


def lgb_stacking_classifier(train_base, test_base, folds=5):
    # Random Search for 2nd level model
    lgb_params = {"boosting": "gbdt",
                  "objective": "multiclass",
                  "num_class": 3,
                  "metric": "multi_logloss",
                  "num_leaves": 64,
                  "max_depth": 4,
                  "learning_rate": 0.04,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.9,
                  "feature_fraction": 0.9,
                  "reg_alpha": 0,
                  "reg_lambda": 0.02,
                  "n_jobs": -1,
                  "n_estimators": 3000,
                  "verbose": -1}
    lgb_score, lgb_importances, lgb_oof, lgb_pred = lgb_clf_training(train_base,
                                                                     test_base,
                                                                     num_folds=folds,
                                                                     params=lgb_params,
                                                                     stratified=False,
                                                                     early_stop_rounds=100)
    return [lgb_score, lgb_importances, lgb_oof, lgb_pred]


def xgb_stacking_classifier(train_base, test_base, folds=5):
    # Random Search for 2nd level model
    xgb_params = {"n_estimators": 5000,
                  "max_depth": 5,
                  "learning_rate": 0.05,
                  "objective ": "multi:softmax",
                  "num_class": 3,
                  "booster": "gbtree",
                  "subsample":0.95,
                  "colsample_bytree": 1,
                  "colsample_bylevel": 1,
                  "colsample_bynode": 1,
                  "n_jobs": mp.cpu_count(),
                  "reg_alpha": 0, 
                  "reg_lambda": 0.03,
                  "verbosity": 0}
    xgb_score, xgb_importances, xgb_oof, xgb_pred = xgb_clf_training(train_base,
                                                                     test_base,
                                                                     num_folds=folds,
                                                                     params=xgb_params,
                                                                     stratified=False,
                                                                     early_stop_rounds=100)
    return [xgb_score, xgb_importances, xgb_oof, xgb_pred]


def lgb_base_training(folds=5, embedding_input=None, embedding_enable=True,
                      train_mode="cv", train_iters=240):
    """Base submission training."""
    train_base = load_data("train_feature_lgb.pkl")
    test_base = load_data("test_feature_lgb.pkl")

    if embedding_enable:
        train_embedding, test_embedding = embedding_input[0], embedding_input[1]
        train_base = pd.merge(train_base, train_embedding, on="boat_id", how="left")
        test_base = pd.merge(test_base, test_embedding, on="boat_id", how="left")

    # Training lightgbm
    print("@Lightgbm training: ")
    print("=================================")
    lgb_params = {"boosting_type": "gbdt",       # "boosting": "gbdt"
                  "objective": "multiclass",
                  "num_class": 3,
                  "metric": "multi_logloss",
                  "num_leaves": 128,
                  "max_depth": 6,
                  "learning_rate": 0.07,
                  "subsample_freq": 1,          # "bagging_freq": 1
                  "subsample": 0.9,             # "bagging_fraction": 0.9
                  "colsample_bytree": 1,        # "feature_fraction": 1
                  "reg_alpha": 0,
                  "reg_lambda": 0.02,
                  "n_jobs": -1,
                  "n_estimators": 5000,
                  "verbose": -1}
    lgb_score, lgb_importances, lgb_oof, lgb_pred = lgb_clf_training(train_base,
                                                                     test_base,
                                                                     num_folds=folds,
                                                                     params=lgb_params,
                                                                     stratified=True,
                                                                     early_stop_rounds=100,
                                                                     mode=train_mode,
                                                                     mode_iters=train_iters)
    print("@Training distribution: Class 0: {:.4f}, Class 1: {:.4f}, Class 2: {:.4f}".format(
        len(train_base.query("target == 0")) / len(train_base),
        len(train_base.query("target == 1")) / len(train_base),
        len(train_base.query("target == 2")) / len(train_base)))
    print("=================================")
    return [lgb_score, lgb_importances, lgb_oof, lgb_pred]


def xgb_base_training(folds=5, embedding_input=None, embedding_enable=True,
                     train_mode="cv", train_iters=270):
    """Base submission training."""
    train_base = load_data("train_feature_xgb.pkl")
    test_base = load_data("test_feature_xgb.pkl")

    if embedding_enable:
        train_embedding, test_embedding = embedding_input[0], embedding_input[1]
        train_base = pd.merge(train_base, train_embedding, on="boat_id", how="left")
        test_base = pd.merge(test_base, test_embedding, on="boat_id", how="left")

    # Training xgboost
    print("@XGBoost training: ")
    print("=================================")
    xgb_params = {"n_estimators": 5000,
                  "max_depth": 6,
                  "learning_rate": 0.08,
                  "objective ": "multi:softmax",
                  "num_class": 3,
                  "booster": "gbtree",
                  "subsample":0.93,
                  "colsample_bytree": 1,
                  "colsample_bylevel": 1,
                  "colsample_bynode": 1,
                  "n_jobs": mp.cpu_count(),
                  "reg_alpha": 0, 
                  "reg_lambda": 0.03,
                  "verbosity": 0}
    xgb_score, xgb_importances, xgb_oof, xgb_pred = xgb_clf_training(train_base,
                                                                     test_base,
                                                                     num_folds=folds,
                                                                     params=xgb_params,
                                                                     stratified=False,
                                                                     early_stop_rounds=100,
                                                                     mode=train_mode,
                                                                     mode_iters=train_iters)
    print("@Training distribution: Class 0: {:.4f}, Class 1: {:.4f}, Class 2: {:.4f}".format(
        len(train_base.query("target == 0")) / len(train_base),
        len(train_base.query("target == 1")) / len(train_base),
        len(train_base.query("target == 2")) / len(train_base)))
    print("=================================")
    return [xgb_score, xgb_importances, xgb_oof, xgb_pred]


def lgb_clf_embedding_list_train(folds=5, id_list=None, embedding_enable=True,
                                 train_mode="cv", train_iters=240):
    """Training the lgb classifier using embedding list."""
    train_embedding_list = load_data("train_embedding_cbow_list.pkl")
    test_embedding_list = load_data("test_embedding_cbow_list.pkl")

    if id_list == None:
        id_list = [0]
    else:
        train_embedding_list = [train_embedding_list[i] for i in id_list]
        test_embedding_list = [test_embedding_list[i] for i in id_list]

    res_list = []
    for train_embedding, test_embedding in zip(train_embedding_list, test_embedding_list):
        embedding = [train_embedding, test_embedding]
        res = lgb_base_training(folds=folds, embedding_input=embedding,
                                embedding_enable=embedding_enable,
                                train_mode=train_mode, train_iters=train_iters)
        res_list.append(res)
        res[2].to_csv(".//tcdata_tmp//lgb_oof.csv", index=False)
    return res_list


def xgb_clf_embedding_list_train(folds=5, id_list=None, embedding_enable=True,
                                 train_mode="cv", train_iters=270):
    """Training the xgb classifier using embedding list."""
    train_embedding_list = load_data("train_embedding_cbow_list.pkl")
    test_embedding_list = load_data("test_embedding_cbow_list.pkl")

    if id_list == None:
        id_list = [0]
    else:
        train_embedding_list = [train_embedding_list[i] for i in id_list]
        test_embedding_list = [test_embedding_list[i] for i in id_list]

    res_list = []
    for train_embedding, test_embedding in zip(train_embedding_list, test_embedding_list):
        embedding = [train_embedding, test_embedding]
        res = xgb_base_training(folds=folds, embedding_input=embedding,
                                embedding_enable=embedding_enable,
                                train_mode=train_mode, train_iters=train_iters)
        res_list.append(res)
    return res_list


def stack_clf_embedding_list_training(res_list=None, folds=5, stacking_method="lgb"):
    """Stacking for ehancement of the robustness."""
    X_train = pd.DataFrame(np.hstack([df_list[2][["pred_0", "pred_1", "pred_2"]].values for df_list in res_list]))
    X_test = pd.DataFrame(np.hstack([df_list[3][["pred_0", "pred_1", "pred_2"]].values for df_list in res_list]))
    X_train["boat_id"], X_test["boat_id"] = res_list[0][2]["boat_id"].values, res_list[0][3]["boat_id"].values
    target = load_data("train_target.pkl")
    X_train = pd.merge(X_train, target, on="boat_id", how="left")

    if stacking_method == "lgb":
        print("\n@LGB stacking of results:")
        print("-----------------------------")
        stacking_pred = lgb_stacking_classifier(X_train, X_test, folds=folds)[-1]
        print("-----------------------------")
    elif stacking_method == "xgb":
        print("\n@XGB stacking of results:")
        print("-----------------------------")
        stacking_pred = xgb_stacking_classifier(X_train, X_test, folds=folds)[-1]
        print("-----------------------------")
    else:
        from sklearn.metrics import accuracy_score, f1_score
        stacking_valid = np.mean([df_list[2][["pred_0", "pred_1", "pred_2"]].values for df_list in res_list], axis=0)
        stacking_pred = np.mean([df_list[3][["pred_0", "pred_1", "pred_2"]].values for df_list in res_list], axis=0)

        print("\n@Mean of results:")
        print("-----------------------------")
        print("-- CV Accuracy score: {:.5f}".format(
            accuracy_score(target["target"].values, np.argmax(stacking_valid, axis=1))))
        print("-- CV F1 score: {:.5f}".format(
            f1_score(target["target"].values, np.argmax(stacking_valid, axis=1),
                     average="macro")))
        print("-----------------------------")
        stacking_pred = pd.DataFrame(stacking_pred, columns=["pred_0", "pred_1", "pred_2"])
        stacking_pred["boat_id"] = X_test["boat_id"].values

    return stacking_pred


def traj_data_embedding_list_training(lgb_folds=5, xgb_folds=5, stacking_folds=5,
                                      id_list=None, stacking_method="lgb",
                                      xgb_enable=True, lgb_enable=True,
                                      lgb_embedding=True, xgb_embedding=True,
                                      lgb_mode="cv", xgb_mode="cv",
                                      lgb_mode_iters=250, xgb_mode_iters=270):
    """Training the model using the lgb and xgb."""
    if lgb_enable:
        lgb_res_list = lgb_clf_embedding_list_train(folds=lgb_folds,
                                                    id_list=id_list,
                                                    embedding_enable=lgb_embedding,
                                                    train_mode=lgb_mode,
                                                    train_iters=lgb_mode_iters)
    else:
        lgb_res_list = []

    if xgb_enable:
        xgb_res_list = xgb_clf_embedding_list_train(folds=xgb_folds,
                                                    id_list=id_list,
                                                    embedding_enable=xgb_embedding,
                                                    train_mode=xgb_mode,
                                                    train_iters=xgb_mode_iters)
    else:
        xgb_res_list = []

    # Stacking for the enhancement of the submissions
    res_list = lgb_res_list + xgb_res_list
    stacking_pred = stack_clf_embedding_list_training(res_list, folds=stacking_folds,
                                                     
 stacking_method=stacking_method)

    # Preparing the submissions
    pred_to_submission(stacking_pred)
    return None


def pred_to_submission(y_pred=None):
    """Save the submission/y_valid/y_pred results."""
    # The submission ind.
    submission = pd.DataFrame(None)
    submission["boat_id"] = y_pred["boat_id"].astype(int).values
    submission["type"] = np.argmax(
        y_pred[["pred_0", "pred_1", "pred_2"]].values, axis=1)

    #Basic submission stat
    print("\n@Pred to submission:")
    print("-----------------------------")
    print("-- Class 0: {:.5f}, Class 1: {:.5f}, Class 2: {:.5f}".format(
        len(submission.query("type == 0"))/len(submission),
        len(submission.query("type == 1"))/len(submission),
        len(submission.query("type == 2"))/len(submission)))
    print("-- Submission save successed!")
    print("-----------------------------\n")

    submission["type"].replace({0: "拖网", 1: "围网", 2: "刺网"}, inplace=True)
    print()
    submission.to_csv("result.csv", header=False, index=False, encoding="utf-8")


if __name__ == "__main__":
#    # Lightgbm embedding list training
    id_list = [0, 1]

    # Lightgbm training
    lgb_res_list = lgb_clf_embedding_list_train(folds=5, id_list=id_list,
                                                embedding_enable=True,
                                                train_mode="cv")
    df = [training_res_to_log(training_res=lgb_res_list[i][0],
                              comment="embedding lgb {}".format(i)) for i in range(len(id_list))]

#    # XGBoost training
#    xgb_res_list = xgb_clf_embedding_list_train(folds=5, id_list=id_list,
#                                                embedding_enable=True,
#                                                train_mode="cv", train_iters=270)
#    df = [training_res_to_log(training_res=xgb_res_list[i][0],
#                              comment="embedding xgb {}".format(i)) for i in range(len(id_list))]
#
#    # Stacking training
#    stacking_method = "lgb"
#    stacking_pred = stack_clf_embedding_list_training(res_list=lgb_res_list,
#                                                     folds=5, stacking_method=stacking_method)
