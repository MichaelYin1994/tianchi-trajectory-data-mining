#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:17:50 2020

@author: yinzhuo
"""

import os
import warnings
from functools import partial
import gc
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from utils import LoadSave, timefn
from gensim.models import word2vec
from tqdm import tqdm
import multiprocessing as mp

np.random.seed(9102)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def load_data(name=None):
    """Load data from .//tcdata_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//tcdata_tmp//" + name)
    return data


def traj_to_bin(traj=None, x_min=12031967.16239096, x_max=14226964.881853,
                y_min=1623579.449434373, y_max=4689471.1780792,
                row_bins=3832, col_bins=2743):
    # col_bins = (14226964.881853 - 12031967.16239096) / 600
    # row_bins = (4689471.1780792 - 1623579.449434373) / 600
    # Establish bins on x direction and y direction
    x_bins = np.linspace(x_min, x_max, endpoint=True, num=col_bins + 1)
    y_bins = np.linspace(y_min, y_max, endpoint=True, num=row_bins + 1)

    # Determine each x coordinate belong to which bin
    traj.sort_values(by='x', inplace=True)
    x_res = np.zeros((len(traj), ))
    j = 0
    for i in range(1, col_bins + 1):
        low, high = x_bins[i-1], x_bins[i]
        while( j < len(traj)):
            # low - 0.001 for numeric stable.
            if (traj["x"].iloc[j] <= high) & (traj["x"].iloc[j] > low - 0.001):
                x_res[j] = i
                j += 1
            else:
                break
    traj["x_grid"] = x_res
    traj["x_grid"] = traj["x_grid"].astype(int)
    traj["x_grid"] = traj["x_grid"].apply(str)

    # Determine each y coordinate belong to which bin
    traj.sort_values(by='y', inplace=True)
    y_res = np.zeros((len(traj), ))
    j = 0
    for i in range(1, row_bins + 1):
        low, high = y_bins[i-1], y_bins[i]
        while( j < len(traj)):
            # low - 0.001 for numeric stable.
            if (traj["y"].iloc[j] <= high) & (traj["y"].iloc[j] > low - 0.001):
                y_res[j] = i
                j += 1
            else:
                break
    traj["y_grid"] = y_res
    traj["y_grid"] = traj["y_grid"].astype(int)
    traj["y_grid"] = traj["y_grid"].apply(str)

    # Determine which bin each coordinate belongs to.
    traj["no_bin"] = [i + "_" + j for i, j in zip(
        traj["x_grid"].values.tolist(), traj["y_grid"].values.tolist())]
    if "time" in traj.columns:
        traj.sort_values(by='time', inplace=True)
    return traj


@timefn
def preparing_traj_data_corpus(bin_size=600):
    """Preparing the training corpus for the traj2vec model."""
    # Loading all the data
    train_data = load_data("train_semantic.pkl")
    test_data = load_data("test_semantic.pkl")
    ais_data = load_data("ais_semantic.pkl")

    train_concat = load_data("train_semantic_concat.pkl")
    test_concat = load_data("test_semantic_concat.pkl")

    # Print statistics
    x_min = min(train_concat["x"].min(), test_concat["x"].min())
    x_max = max(train_concat["x"].max(), test_concat["x"].max())
    y_min = min(train_concat["y"].min(), test_concat["y"].min())
    y_max = max(train_concat["y"].max(), test_concat["y"].max())

    col_bins = int((x_max - x_min) / bin_size)
    row_bins = int((y_max - y_min) / bin_size)

    # Start cutting the traj to bins
    traj_total = train_data + test_data + ais_data
    res = []

    # Multi-processing for loop.
    partial_work = partial(traj_to_bin, col_bins=col_bins, row_bins=row_bins,
                           x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    with mp.Pool(processes=mp.cpu_count()) as p:
        res = list(tqdm(p.imap(partial_work, traj_total),
                        total=len(traj_total)))

    unique_words = [traj["no_bin"].nunique() for traj in res]
    print("\n@Cutting results basic stat:")
    print("-----------------------------")
    print("@Mean uniques: {:.5f}, max: {}, median: {:.5f}, std: {:.5f}".format(
        np.mean(unique_words), np.max(unique_words),
        np.median(unique_words), np.std(unique_words)))
    print("-----------------------------\n")
    file_processor = LoadSave()
    file_processor.save_data(
        path=".//tcdata_tmp//traj_data_corpus.pkl", data=res)


@timefn
def traj_cbow_embedding(traj_data_corpus=None, embedding_size=70,
                        iters=40, min_count=3, window_size=25,
                        seed=9012, num_runs=5, word_feat="no_bin"):
    """CBOW embedding for trajectory data."""
    boat_id = [traj["boat_id"].unique()[0] for traj in traj_data_corpus]
    sentences, embedding_df_list, embedding_model_list = [], [], []
    for traj in traj_data_corpus:
        sentences.append(traj[word_feat].values.tolist())

    print("\n@Start CBOW word embedding at {}".format(datetime.now()))
    print("-------------------------------------------")
    for i in tqdm(range(num_runs)):
        model = word2vec.Word2Vec(sentences, size=embedding_size,
                                  min_count=min_count,
                                  workers=mp.cpu_count(),
                                  window=window_size,
                                  seed=seed, iter=iters, sg=0)

        # Sentance vector
        embedding_vec = []
        for ind, seq in enumerate(sentences):
            seq_vec, word_count = 0, 0
            for word in seq:
                if word not in model:
                    continue
                else:
                    seq_vec += model[word]
                    word_count += 1
            if word_count == 0:
                embedding_vec.append(embedding_size * [0])
            else:
                embedding_vec.append(seq_vec / word_count)
        embedding_vec = np.array(embedding_vec)
        embedding_cbow_df = pd.DataFrame(embedding_vec, 
            columns=["embedding_cbow_{}_{}".format(word_feat, i) for i in range(embedding_size)])
        embedding_cbow_df["boat_id"] = boat_id
        embedding_df_list.append(embedding_cbow_df)
        embedding_model_list.append(model)
    print("-------------------------------------------")
    print("@End CBOW word embedding at {}".format(datetime.now()))
    return embedding_df_list, embedding_model_list


def traj_data_cbow_embedding_generating(embedding_size=70, iters=70,
                                        min_count=3, window_size=25, num_runs=1):
    traj_corpus = load_data("traj_data_corpus.pkl")
    train_nums = len(sorted(os.listdir(".//tcdata//hy_round2_train_20200225//"),
        key=lambda s: int(s.split(".")[0])))
    test_nums = len(sorted(os.listdir(".//tcdata//hy_round2_testA_20200225//"),
        key=lambda s: int(s.split(".")[0])))
    df_list, model_list = traj_cbow_embedding(traj_corpus,
                                              embedding_size=embedding_size,
                                              iters=iters, min_count=min_count,
                                              window_size=window_size,
                                              seed=9012,
                                              num_runs=num_runs,
                                              word_feat="no_bin")

    train_embedding_df_list = [df.iloc[:train_nums].reset_index(
        drop=True) for df in df_list]
    test_embedding_df_list = [df.iloc[train_nums:(
        train_nums+test_nums)].reset_index(drop=True) for df in df_list]

    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//train_embedding_cbow_list.pkl",
                             data=train_embedding_df_list)
    file_processor.save_data(path=".//tcdata_tmp//test_embedding_cbow_list.pkl",
                             data=test_embedding_df_list)


def embedding_geo_info(window_size=25, embedding_size=70,
                       num_runs=1, iters=70, bin_size=600):
    # Step 1: Cutting the trajectories to corpus
    preparing_traj_data_corpus(bin_size=bin_size)

    # Step 2: Training the CBOW model
    traj_data_cbow_embedding_generating(window_size=window_size,
                                        embedding_size=embedding_size,
                                        num_runs=num_runs, iters=iters)


if __name__ == "__main__":
#    # Step 1: Cutting the trajectories to corpus
#    preparing_traj_data_corpus(bin_size=600)

    # Step 2: Training the CBOW model
    traj_data_cbow_embedding_generating(window_size=25, embedding_size=70,
                                        num_runs=3, iters=70)
