#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:13:44 2020

@author: michael
"""

import gc
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import LoadSave, training_res_to_log
#from utils import feature_selection_info_query
from traj_data_classification import xgb_clf_embedding_list_train
from embedding_geo_info import traj_cbow_embedding

np.random.seed(1080)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def load_data(name=None):
    """Load data from .//tcdata_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//tcdata_tmp//" + name)
    return data


def traj_activity_area_quantile(traj=None, quantile_low=None,
                                quantile_high=None):
    """Activity area of a trajectory"""
    feature_vals = []
    for low, high in zip(quantile_low, quantile_high):
        length = (traj["x"].quantile(high) - traj["x"].quantile(low))
        width = (traj["y"].quantile(high) - traj["y"].quantile(low))
        feature_vals.append(length * width)
    return feature_vals


def traj_top_frequent_cordinates(traj=None):
    """Top frequent coordinates and its frequency."""
    gp = traj.groupby(["lon", "lat"]).count()
    total_uniques, unique_percentage = [len(gp)], [len(gp) / len(traj)]
    gp.reset_index(inplace=True)
    
    array_tmp = gp.sort_values(by="boat_id", ascending=False)[["lon", "lat"]].values
    coord_freq = array_tmp[0].tolist()
    return total_uniques + unique_percentage + coord_freq


def traj_quantile_features(traj, quantile=[0.1, 0.15, 0.25],
                           feature_name="x"):
    """Quantile statistics for a specified feature."""
    if len(traj) == 0:
        return [0] * len(quantile)
    feature_vals = []
    for qu in quantile:
        feature_vals.append(traj[feature_name].quantile(qu))
    return feature_vals


def traj_coordiate_divergence(traj=None, quantile=None):
    """The divergence of the trajectory coordiates."""
    quantile = quantile or [0.5, 0.75, 0.99]
    if len(traj) == 0:
        return [0] * len(quantile)

    mean_coord = traj[["x", "y"]].mean(axis=0).values
    dist_array = traj[["x", "y"]].values - mean_coord
    dist_array = np.sqrt(dist_array[:, 0]**2 + dist_array[:, 1]**2)

    ret_res = []
    for qu in quantile:
        ret_res.append(np.quantile(dist_array, q=qu))
    return ret_res


def traj_frequent_divergence(traj=None, quantile=None):
    """The divergence of the frequent coordinates."""
    quantile = quantile or [0.5, 0.75, 0.99]
    if len(traj) == 0:
        return [0] * len(quantile)

    gp = traj.groupby(["x", "y"]).count().reset_index()
    freq_coord = gp.sort_values(
        by="boat_id", ascending=False)[["x", "y"]].values[0].tolist()
    dist_array = traj[["x", "y"]].values - freq_coord
    dist_array = np.sqrt(dist_array[:, 0]**2 + dist_array[:, 1]**2)

    ret_res = []
    for qu in quantile:
        ret_res.append(np.quantile(dist_array, q=qu))
    return ret_res


def percentile_ad(n):
    """The absolute_deviation according to percentile(n)."""
    """https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function"""
    def percentile_absolute_deviation(x, scale=1.4826):
        val_tmp = np.percentile(x, n)
        return np.median(np.abs(x - val_tmp)) * scale
    percentile_absolute_deviation.__name__ = 'percentile_ad_%s' % n
    return percentile_absolute_deviation


def percentile(n):
    """https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function"""
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

########################## Statistical features part ##########################
def traj_coord_stat(traj=None):
    """Feature engineering: coordinates"""
    traj["y/x"] = traj["y"] / traj["x"]
    traj["x2_y2"] = np.sqrt((traj["x"] / 10000)**2 + (traj["y"] / 10000)**2)
    traj["lat/lon"] = traj["lat"] / traj["lon"]
    traj["lon-lat"] = traj["lon"] - traj["lat"]
    traj["lon2_lat2"] = np.sqrt((traj["lon"])**2 + (traj["lat"])**2)
    traj_move = traj[traj["is_stop"] == -1]
    traj_stop = traj[traj["is_stop"] != -1]
    feature_vals, feature_names = [], []

    # Step 1: Basic statistics
    stat_feature_fcn = [np.median, np.ptp, np.std]
    for col_name in ["lon", "lat", "y/x"]:
        for fcn in stat_feature_fcn:
            feature_names.append("base_{}_{}".format(col_name, fcn.__name__))
            feature_vals.append(fcn(traj[col_name]))
        feature_vals.append(traj[col_name].iloc[0])
        feature_vals.append(traj[col_name].iloc[-1])
        feature_names.append("base_{}_first".format(col_name))
        feature_names.append("base_{}_last".format(col_name))

    stat_feature_fcn = [np.std, np.median]
    for col_name in ["lon", "lat", "y/x"]:
        for fcn in stat_feature_fcn:
            feature_names.append("stop_{}_{}".format(col_name, fcn.__name__))
            if len(traj_stop) == 0:
                feature_vals.append(np.nan)
            else:
                feature_vals.append(fcn(traj_stop[col_name]))

    # Step 4: The most frequent coordinates and its frequency
    feature_names.extend(["base_total_uniques", "base_dup_precent",
                          "base_most_freq_lon", "base_most_freq_lat"])
    feature_vals.extend(traj_top_frequent_cordinates(traj))

    # Step 5: Quantile features
    quantile = np.linspace(0.02, 0.99, 25)
    feature_names.extend(["base_lon2_lat2_quantile_{}".format(i) for i in quantile])
    feature_vals.extend(traj_quantile_features(traj, quantile=quantile,
                                               feature_name="lon2_lat2"))

#    quantile = np.linspace(0, 1, 2)
#    feature_names.extend(["base_y/x_quantile_{}".format(i) for i in quantile])
#    feature_vals.extend(traj_quantile_features(traj_move, quantile=quantile,
#                                               feature_name="y/x"))

    # quantile = np.linspace(0, 1, 14)
    quantile = np.linspace(0.02, 0.99, 19)
    feature_names.extend(["base_lon-lat_quantile_{}".format(i) for i in quantile])
    feature_vals.extend(traj_quantile_features(traj_move, quantile=quantile,
                                               feature_name="lon-lat"))

    quantile = np.linspace(0.02, 0.99, 19)
    feature_names.extend(["base_y/x_quantile_{}".format(i) for i in quantile])
    feature_vals.extend(traj_quantile_features(traj_move, quantile=quantile,
                                               feature_name="y/x"))

    df = pd.DataFrame(np.array(feature_vals).reshape((1, -1)),
                      columns=feature_names)
    return df


def traj_speed_stat(traj=None):
    """Feature engineering: speed"""
    traj_move = traj[traj["is_stop"] == -1]
#    traj[traj["is_stop"] != -1]["speed"] = 0
    feature_vals, feature_names = [], []

    # Step 1: Basic statistics
    stat_feature_fcn = [np.mean, np.std, np.median]
    for col_name in ["speed"]:
        for fcn in stat_feature_fcn:
            feature_names.append("move_{}_{}".format(col_name, fcn.__name__))
            if len(traj_move) != 0:
                feature_vals.append(fcn(traj_move[col_name]))
            else:
                feature_vals.append(0)

    # Step 1: Quantile features(0.02, 0.98, 43)
    quantile = np.linspace(0.02, 0.99, 43)
    feature_names.extend(["base_speed_quantile_{}".format(i) for i in quantile])
    feature_vals.extend(traj_quantile_features(traj_move, quantile=quantile,
                                               feature_name="speed"))

    # Step 2: Histogram of speed(15, (0, 6))
    num_bins, bin_range = 10, (0, 6)
    hist_vals, hist_cut = np.histogram(traj_move["speed"].values, bins=num_bins,
                                       range=bin_range, normed=True)
    feature_names.extend([
        "base_speed_hist_{}".format(i) for i in hist_cut[1:]])
    feature_vals.extend(hist_vals.tolist())

    # Step 3: Kurt and skew features
    feature_vals.extend([traj_move["speed"].kurt(), traj_move["speed"].skew()])
    feature_names.extend(["move_speed_kurt", "move_speed_skew"])

    df = pd.DataFrame(np.array(feature_vals).reshape((1, -1)),
                      columns=feature_names)
    return df


def traj_dir_stat(traj=None):
    """Feature engineering: direction"""
    traj_move = traj[traj["is_stop"] == -1]
    traj_stop = traj[traj["is_stop"] != -1]
    feature_vals, feature_names = [], []

#    # Step 1: Quantile features([0.001, 0.1, 15])
#    quantile = np.linspace(0.001, 0.1, 15)
#    feature_names.extend(["base_dir_quantile_{}".format(i) for i in quantile])
#    feature_vals.extend(traj_quantile_features(traj, quantile=quantile,
#                                               feature_name="direction"))

    # Step 2: Histogram of speed(10, (0, 15))
    num_bins, bin_range = 8, (0, 15)
    hist_vals, hist_cut = np.histogram(traj_move["direction"].values, bins=num_bins,
                                       range=bin_range, normed=True)
    feature_names.extend(["base_dir_hist_{}".format(
        i) for i in hist_cut[1:]])
    feature_vals.extend(hist_vals.tolist())

    df = pd.DataFrame(np.array(feature_vals).reshape((1, -1)),
                      columns=feature_names)
    return df


def traj_expert(traj=None):
    """Feature engineering: expert features"""
    traj_move = traj[traj["is_stop"] == -1]
    traj_stop = traj[traj["is_stop"] != -1]
    feature_vals, feature_names = [], []

    # Step 1: Coordinate divergence features[0.01, 0.99, 14]
    quantile = np.linspace(0.01, 0.99, 14).tolist()
    feature_names.extend(["base_expert_coord_div_quantile_{}".format(
        i) for i in quantile])
    feature_vals.extend(traj_coordiate_divergence(traj, quantile=quantile))

    # Step 2: Frequent coordinate divergence.
    quantile = np.linspace(0.01, 0.99, 14).tolist()
    feature_names.extend(["base_expert_freq_coord_div_quantile_{}".format(
        i) for i in quantile])
    feature_vals.extend(traj_frequent_divergence(traj, quantile=quantile))

#    # Step 1: Coordinate divergence features[0.01, 0.99, 14]
#    quantile = np.linspace(0.01, 0.99, 18).tolist()
#    feature_names.extend(["base_expert_coord_div_quantile_{}".format(
#        i) for i in quantile])
#    feature_vals.extend(traj_coordiate_divergence(traj, quantile=quantile))
#
#    # Step 2: Frequent coordinate divergence.
#    quantile = np.linspace(0.01, 0.15, 6).tolist()
#    feature_names.extend(["base_expert_freq_coord_div_quantile_{}".format(
#        i) for i in quantile])
#    feature_vals.extend(traj_frequent_divergence(traj, quantile=quantile))

    df = pd.DataFrame(np.array(feature_vals).reshape((1, -1)),
                      columns=feature_names)
    return df    


############################# Text features part #############################
def traj_data_signal_embedding():
    """Loading the embedding vectors."""
    file_processor = LoadSave()
    train_embedding = file_processor.load_data(path=".//tcdata_tmp//train_signal_embedding.pkl")
    test_embedding = file_processor.load_data(path=".//tcdata_tmp//test_signal_embedding.pkl")
    return pd.concat([train_embedding, test_embedding], axis=0, ignore_index=True)


def traj_data_direction_embedding(total_data=None, embedding_size=15,
                                  iters=70, window_size=20, min_count=5):
    """Embedding of the direction feature of each trajectory."""
    total_data_corpus = []
    for traj in total_data:
        traj["direction_str"] = traj["direction"].apply(str)
#        traj["direction_str"][traj["is_stop"] != -1] = "0"
        total_data_corpus.append(traj)
    df_list, model_list = traj_cbow_embedding(total_data_corpus, embedding_size=embedding_size,
                                              iters=iters, min_count=min_count,
                                              window_size=window_size, seed=9012,
                                              num_runs=1, word_feat="direction_str")
    dir_embedding = df_list[0].reset_index(drop=True)
    return dir_embedding


def traj_data_speed_embedding(total_data=None, embedding_size=15,
                              iters=70, window_size=20, min_count=5):
    """Embedding of the speed feature of each trajectory."""
    total_data_corpus = []
    for traj in total_data:
        traj["speed_str"] = traj["speed"].apply(str)
#        traj["speed_str"][traj["is_stop"] != -1] = "0"
        total_data_corpus.append(traj)
    df_list, model_list = traj_cbow_embedding(total_data, embedding_size=embedding_size,
                                              iters=iters, min_count=min_count,
                                              window_size=window_size, seed=9012,
                                              num_runs=1, word_feat="speed_str")
    speed_embedding = df_list[0].reset_index(drop=True)
    return speed_embedding


def traj_data_speed_tfidf(total_data=None, max_features=10):
    """TF-IDF features for the speed feature."""
    boat_id = [traj["boat_id"].unique()[0] for traj in total_data]

    total_data_corpus = []
    for traj in total_data:
        traj["speed_str"] = traj["speed"].apply(lambda x: str(x*100))
        traj["speed_str"][traj["is_stop"] != -1] = "0"
        total_data_corpus.append(" ".join(traj["speed_str"].values.tolist()))

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None,
                                 max_features=max_features)
    total_features = vectorizer.fit_transform(total_data_corpus).toarray()
    total_features = pd.DataFrame(total_features, columns=["speed_tfidf_{}".format(
        i) for i in range(total_features.shape[1])])
    total_features["boat_id"] = boat_id
    return total_features


def traj_data_bin_tfidf(total_data=None, max_features=10):
    """TF-IDF features for the bin."""
    boat_id = [traj["boat_id"].unique()[0] for traj in total_data]

    traj_corpus = load_data("traj_data_corpus.pkl")
    traj_data_corpus = []
    for traj in traj_corpus:
        traj_data_corpus.append(" ".join(traj["no_bin"].values.tolist()))

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l1",
                                 max_features=max_features)
    total_features = vectorizer.fit_transform(traj_data_corpus).toarray()
    total_features = pd.DataFrame(total_features, columns=["bin_tfidf_{}".format(
        i) for i in range(total_features.shape[1])])
    total_features["boat_id"] = boat_id
    return total_features

############################# Training part #############################
def stat_feature_engineering_xgb():
    train_data = load_data("train_semantic.pkl")
    test_data_a = load_data("test_semantic.pkl")
    train_nums = len(train_data)

    total_data = train_data + test_data_a
    boat_id = [traj["boat_id"].unique()[0] for traj in total_data]
    labels = [traj["type"].unique()[0] for traj in train_data]
    total_features = pd.DataFrame(None)
    total_features["boat_id"] = boat_id

    # Step 1: coordinate stat features.
    with mp.Pool(processes = mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(traj_coord_stat, total_data),
                        total=len(total_data)))
    coord_features = pd.concat(tmp, axis=0, ignore_index=True)
    coord_features["boat_id"] = boat_id
    total_features = pd.merge(total_features, coord_features,
                              on="boat_id", how="left")


    # Step 2: speed stat features.
    with mp.Pool(processes = mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(traj_speed_stat, total_data),
                        total=len(total_data)))
    speed_features = pd.concat(tmp, axis=0, ignore_index=True)
    speed_features["boat_id"] = boat_id
    total_features = pd.merge(total_features, speed_features,
                              on="boat_id", how="left")


    # Step 4: expert features.
    with mp.Pool(processes = mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(traj_expert, total_data),
                        total=len(total_data)))
    expert_features = pd.concat(tmp, axis=0, ignore_index=True)
    expert_features["boat_id"] = boat_id
    total_features = pd.merge(total_features, expert_features,
                              on="boat_id", how="left")


    # Step 5: Concat the speed_dir embedding vector
    dir_embedding = traj_data_direction_embedding(total_data, embedding_size=8,
                                                  iters=70, window_size=20,
                                                  min_count=3)
    total_features = pd.merge(total_features, dir_embedding, on="boat_id",
                              how="left")

    speed_embedding = traj_data_speed_embedding(total_data, embedding_size=10,
                                                iters=70, window_size=20,
                                                min_count=3)
    total_features = pd.merge(total_features, speed_embedding, on="boat_id",
                              how="left")


#    # Step 7: speed tfidf
#    speed_tfidf = traj_data_speed_tfidf(total_data, max_features=40)
#    total_features = pd.merge(total_features, speed_tfidf, on="boat_id",
#                              how="left")

#    # Step 8: GEO tfidf
#    bin_tfidf = traj_data_bin_tfidf(total_data, max_features=70)
#    total_features = pd.merge(total_features, bin_tfidf, on="boat_id",
#                              how="left")

    ##################################################
    train_feature = total_features.iloc[:train_nums].reset_index(drop=True).copy()
    test_feature = total_features.iloc[train_nums:].reset_index(drop=True).copy()
    train_feature["target"] = labels

    print("\n--Train samples: {}, testA samples: {}.".format(
        len(train_feature), len(test_feature)))
    print("--Train cols: {}, test cols: {}.".format(
        train_feature.shape[1], test_feature.shape[1]))
    print("--Unique train cols: {}, unique testA cols: {}.\n".format(
        len(np.unique(train_feature.columns)),
        len(np.unique(test_feature.columns))))
    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//train_feature_xgb.pkl",
                             data=train_feature)
    file_processor.save_data(path=".//tcdata_tmp//train_target.pkl",
                             data=train_feature[["boat_id", "target"]])
    file_processor.save_data(path=".//tcdata_tmp//test_feature_xgb.pkl",
                             data=test_feature)
    gc.collect()


if __name__ == "__main__":
    train_data = load_data("train_semantic.pkl")
    test_data_a = load_data("test_semantic.pkl")
    train_nums = len(train_data)

    total_data = train_data + test_data_a
    boat_id = [traj["boat_id"].unique()[0] for traj in total_data]
    labels = [traj["type"].unique()[0] for traj in train_data]
    total_features = pd.DataFrame(None)
    total_features["boat_id"] = boat_id
    traj = train_data[120]


    # Step 1: coordinate stat features.
    with mp.Pool(processes = mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(traj_coord_stat, total_data),
                        total=len(total_data)))
    coord_features = pd.concat(tmp, axis=0, ignore_index=True)
    coord_features["boat_id"] = boat_id
    total_features = pd.merge(total_features, coord_features,
                              on="boat_id", how="left")


    # Step 2: speed stat features.
    with mp.Pool(processes = mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(traj_speed_stat, total_data),
                        total=len(total_data)))
    speed_features = pd.concat(tmp, axis=0, ignore_index=True)
    speed_features["boat_id"] = boat_id
    total_features = pd.merge(total_features, speed_features,
                              on="boat_id", how="left")


    # Step 4: expert features.
    with mp.Pool(processes = mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(traj_expert, total_data),
                        total=len(total_data)))
    expert_features = pd.concat(tmp, axis=0, ignore_index=True)
    expert_features["boat_id"] = boat_id
    total_features = pd.merge(total_features, expert_features,
                              on="boat_id", how="left")


    # Step 5: Concat the speed_dir embedding vector
    dir_embedding = traj_data_direction_embedding(total_data, embedding_size=8,
                                                  iters=70, window_size=20,
                                                  min_count=3)
    total_features = pd.merge(total_features, dir_embedding, on="boat_id",
                              how="left")

    speed_embedding = traj_data_speed_embedding(total_data, embedding_size=10,
                                                iters=70, window_size=20,
                                                min_count=3)
    total_features = pd.merge(total_features, speed_embedding, on="boat_id",
                              how="left")

#    # Step 7: speed tfidf
#    speed_tfidf = traj_data_speed_tfidf(total_data, max_features=40)
#    total_features = pd.merge(total_features, speed_tfidf, on="boat_id",
#                              how="left")

    ##################################################
    train_feature = total_features.iloc[:train_nums].reset_index(drop=True).copy()
    test_feature = total_features.iloc[train_nums:].reset_index(drop=True).copy()
    train_feature["target"] = labels

    print("\n-- Train samples: {}, testA samples: {}.".format(
        len(train_feature), len(test_feature)))
    print("-- Train cols: {}, test cols: {}.".format(
        train_feature.shape[1], test_feature.shape[1]))
    print("-- Unique train cols: {}, unique testA cols: {}.\n".format(
        len(np.unique(train_feature.columns)),
        len(np.unique(test_feature.columns))))
    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//train_feature_xgb.pkl",
                             data=train_feature)
    file_processor.save_data(path=".//tcdata_tmp//train_target.pkl",
                             data=train_feature[["boat_id", "target"]])
    file_processor.save_data(path=".//tcdata_tmp//test_feature_xgb.pkl",
                             data=test_feature)
    gc.collect()

    for embedding_id in [0, 1, 2]:
        xgb_res_list = xgb_clf_embedding_list_train(folds=5, id_list=[embedding_id],
                                                    embedding_enable=True)
        df = training_res_to_log(training_res=xgb_res_list[0][0], comment="xgb_{}".format(embedding_id))
