#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:50:05 2020

@author: zhuoyin94
"""
import os
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from utils import LoadSave

np.random.seed(2022)
###############################################################################
def read_traj(file_name=None):
    """Reading a sigle trajectory *.csv."""
    PATH = ".//tcdata//hy_round2_train_20200225_local//"
    return pd.read_csv(PATH + file_name, encoding="utf-8")


def traj_data_train_test_index_generation(train_ratio=0.85, n_samples=8166,
                                          target=None, method="random"):
    """Generating the indexes of the training and testing data for indexing."""
    n_train_samples = int(train_ratio * n_samples)

    if method == "random":
        total_index = np.arange(0, n_samples)
        np.random.shuffle(total_index)

        train_index = total_index[:n_train_samples]
        test_index = total_index[n_train_samples:]
    else:
        total_index = np.arange(0, n_samples).reshape((-1, 1))
        sss = StratifiedShuffleSplit(n_splits=5, train_size=train_ratio,
                                     random_state=2022)
        train_index, test_index = [], []

        for train_id, test_id in sss.split(total_index, target.reshape((-1, 1))):
            train_index.append(train_id)
            test_index.append(test_id)
        train_index, test_index = train_index[0], test_index[0]

    return np.sort(train_index), np.sort(test_index)


def traj_data_train_test_split(train_ratio=0.85):
    """Split the training data into training dataset and testing dataset, This
       is for the online docker docker evaluation testing.
    """
    PATH = ".//tcdata//hy_round2_train_20200225_local//"
    file_names = sorted(os.listdir(PATH), key=lambda s: int(s.split(".")[0]))

    print("\n@Read all raw traj data started at: {}".format(datetime.now()))
    print("-----------------------------")
    with mp.Pool(processes = mp.cpu_count()) as p:
        traj_data_total = list(tqdm(p.imap(read_traj, file_names),
                                    total=len(file_names)))
    print("-----------------------------")
    print("@End at: {}".format(datetime.now()))

    # Map the Chinese labels into the numerics
    str_to_label = {"刺网": 2, "围网": 1, "拖网": 0}
    target = [traj["type"].unique()[0] for traj in traj_data_total]
    target = np.array([str_to_label[i] for i in target])

    train_index, test_index = traj_data_train_test_index_generation(
        train_ratio=train_ratio, n_samples=len(traj_data_total),
        target=target, method="stratified")
    traj_data_train = [traj_data_total[i] for i in train_index]
    traj_data_train_fnames = [file_names[i] for i in train_index]
    traj_data_test = [traj_data_total[i] for i in test_index]
    traj_data_test_fnames = [file_names[i] for i in test_index]

    train_target_dist = [target[i] for i in train_index]
    test_target_dist = [target[i] for i in test_index]
    print("@Total target distributions: {}".format(
        np.bincount(target)/len(target)))
    print("@Train distributions: {}".format(
        np.bincount(train_target_dist)/len(traj_data_train)))
    print("@Test distributions: {}".format(
        np.bincount(test_target_dist)/len(traj_data_test)))

    TEST_TARGET_PATH = ".//tcdata_tmp//"
    boat_id = [int(file_names[i].split(".")[0]) for i in test_index]
    df = pd.DataFrame({"boat_id": boat_id, "target": test_target_dist})
    file_processor = LoadSave()
    file_processor.save_data(data=df, path=TEST_TARGET_PATH+"test_target.pkl")

    TRAIN_DATA_PATH = ".//tcdata//hy_round2_train_20200225//"
    file_names = os.listdir(TRAIN_DATA_PATH)
    if len(file_names) != 0:
        raise ValueError("The dir is not empty ! Please remove all file ~~")
    for df, name in zip(traj_data_train, traj_data_train_fnames):
        df.to_csv(TRAIN_DATA_PATH + name, index=False, encoding="utf-8")

    TEST_PATH = ".//tcdata//hy_round2_testA_20200225//"
    file_names = os.listdir(TEST_PATH)
    if len(file_names) != 0:
        raise ValueError("The dir is not empty ! Please remove all files ~~")
    for df, name in zip(traj_data_test, traj_data_test_fnames):
        df.to_csv(TEST_PATH + name, index=False, encoding="utf-8")

    return traj_data_train, traj_data_test


if __name__ == "__main__":
    traj_data_train, traj_data_test = traj_data_train_test_split(train_ratio=0.9995)
