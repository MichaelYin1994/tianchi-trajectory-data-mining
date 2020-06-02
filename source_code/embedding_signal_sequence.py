#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:02:59 2020

@author: zhuoyin94
"""

import warnings
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

def load_train():
    train_data = load_data("train_semantic.pkl")
    return train_data


def load_test():
    test_data = load_data("test_semantic.pkl")
    return test_data


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
        embedding_cbow_df = pd.DataFrame(embedding_vec, columns=[
            "embedding_cbow_{}_{}".format(word_feat, i) for i in range(embedding_size)])
        embedding_cbow_df["boat_id"] = boat_id
        embedding_df_list.append(embedding_cbow_df)
        embedding_model_list.append(model)
    print("-------------------------------------------")
    print("@End CBOW word embedding at {}".format(datetime.now()))
    return embedding_df_list, embedding_model_list


def get_angle_from_coordinate(lat1, long1, lat2, long2):
    """https://stackoverflow.com/questions/3932502/calculate-angle-between-two-latitude-longitude-points"""
    dLon = (long2 - long1)

    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)

    brng = np.arctan2(y, x)

    brng = np.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng
    return brng



def embedding_signal_sequence(speed_embedding=True, dir_embedding=True,
                              speed_dir_embedding=True,
                              speed_filter_stops=False, dir_filter_stops=True):
    """Training the signal embedding."""
    train_data, test_data = load_train(), load_test()

    traj_data_all = train_data + test_data
    train_nums = len(train_data)
    boat_id = [traj["boat_id"].unique()[0] for traj in traj_data_all]
    total_embedding = pd.DataFrame(boat_id, columns=["boat_id"])

    # Step 1: Construct the words
    traj_data_corpus = []
    for traj in traj_data_all:
        traj["speed_str"] = traj["speed"].apply(lambda x: str(int(x*100)))
        traj["direction_str"] = traj["direction"].apply(str)
        if speed_filter_stops:
            traj["speed_str"][traj["is_stop"] != -1] = "0"
        if dir_filter_stops:
            traj["direction_str"][traj["is_stop"] != -1] = "0"

        traj["speed_dir_str"] = traj["speed_str"] + "_" + traj["direction_str"]
        traj_data_corpus.append(traj[["boat_id", "speed_str",
                                      "direction_str", "speed_dir_str"]])

#    traj_data_corpus = []
#    for traj in traj_data_all:
#        lon_val, lat_val = traj["lon"].values, traj["lat"].values
#        angle = get_angle_from_coordinate(lat_val[1:], lon_val[1:],
#                                          lat_val[:-1], lon_val[:-1]).tolist()
#        angle = [angle[0]] + angle
#
#        traj["speed_str"] = traj["speed"].apply(lambda x: str(int(x*100)))
#        traj["direction"] = angle
#        traj["direction_str"] = traj["direction"].apply(str)
#        if speed_filter_stops:
#            traj["speed_str"][traj["is_stop"] != -1] = "0"
#        if dir_filter_stops:
#            traj["direction_str"][traj["is_stop"] != -1] = "0"
#
#        traj["speed_dir_str"] = traj["speed_str"] + "_" + traj["direction_str"]
#        traj_data_corpus.append(traj[["boat_id", "speed_str",
#                                      "direction_str", "speed_dir_str"]])
    
    # Step 2: Training the speed information
    if speed_embedding:
        print("\n@Round 2 speed embedding:")
        print("-----------------------------")
        df_list, model_list = traj_cbow_embedding(traj_data_corpus,
                                                  embedding_size=10,
                                                  iters=40, min_count=3,
                                                  window_size=25, seed=9102,
                                                  num_runs=1, word_feat="speed_str")
        speed_embedding = df_list[0].reset_index(drop=True)
        total_embedding = pd.merge(total_embedding, speed_embedding,
                                   on="boat_id", how="left")
        print("-----------------------------\n")

    # Step 3: Training the direcntion embedding
    if dir_embedding:
        print("\n@Round 2 direction embedding:")
        print("-----------------------------")
        df_list, model_list = traj_cbow_embedding(traj_data_corpus,
                                                  embedding_size=8,
                                                  iters=40, min_count=3,
                                                  window_size=25, seed=9102,
                                                  num_runs=1, word_feat="direction_str")
        dir_embedding = df_list[0].reset_index(drop=True)
        total_embedding = pd.merge(total_embedding, dir_embedding,
                                   on="boat_id", how="left")
        print("-----------------------------\n")

    # Step 4: Training the speed-direcntion embedding
    if speed_dir_embedding:
        print("\n@Round 2 speed_dir embedding:")
        print("-----------------------------")
        df_list, model_list = traj_cbow_embedding(traj_data_corpus,
                                                  embedding_size=12,
                                                  iters=70, min_count=3,
                                                  window_size=25, seed=9102,
                                                  num_runs=1, word_feat="speed_dir_str")
        speed_dir_embedding = df_list[0].reset_index(drop=True)
        total_embedding = pd.merge(total_embedding, speed_dir_embedding,
                                   on="boat_id", how="left")
        print("-----------------------------")

    # Step 5: Svaing the embedding vectorss
    train_embedding = total_embedding.iloc[:train_nums].reset_index(drop=True)
    test_embedding = total_embedding.iloc[train_nums:].reset_index(drop=True)

    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//train_signal_embedding.pkl",
                             data=train_embedding)
    file_processor.save_data(path=".//tcdata_tmp//test_signal_embedding.pkl",
                             data=test_embedding)


if __name__ == "__main__":
    embedding_signal_sequence(speed_embedding=True, dir_embedding=True,
                              speed_dir_embedding=False,
                              speed_filter_stops=False,
                              dir_filter_stops=False)
