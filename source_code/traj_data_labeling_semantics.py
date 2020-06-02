#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:36:08 2020

@author: yinzhuo
"""

import warnings
import json
import multiprocessing as mp
from functools import partial
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from utils import LoadSave, timefn

np.random.seed(2080)
warnings.filterwarnings('ignore')
colors = ["C" + str(i) for i in range(0, 9 + 1)]
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
###############################################################################
def load_pkl_from_path(file_name=None):
    """Load data from .//tcdata_tmp//"""
    file_processor = LoadSave()
    data = file_processor.load_data(path=".//tcdata_tmp//" + file_name)
    return data


def plot_poi_filtered_poi_coordinates(poi=None, poi_filtered=None):
    """Plot mean coordinates of all trajectories."""
    china_geod = gpd.GeoDataFrame.from_file('.//plots//bou2_4p.shp',
                                            encoding='gb18030')

    plt.close("all")
    fig, ax_objs = plt.subplots(1, 2, figsize=(18, 10))
    china_geod.plot(color="k", ax=ax_objs[0])
    china_geod.plot(color="k", ax=ax_objs[1])
    ax_objs[0].scatter(poi["lon"].values, poi["lat"].values, s=9,
           alpha=1, marker=".", color="blue", label="original")
    ax_objs[0].tick_params(axis="both", labelsize=8)
    ax_objs[0].grid(True)
    ax_objs[0].scatter(poi_filtered["lon"].values, poi_filtered["lat"].values,
        s=9, alpha=1, marker=".", color="red", label="REAL")
    ax_objs[0].tick_params(axis="both", labelsize=8)
    ax_objs[0].grid(True)
    ax_objs[0].legend(fontsize=12)

    ax_objs[1].scatter(poi_filtered["lon"].values, poi_filtered["lat"].values,
        ss=9, alpha=1, marker=".", color="red")
    ax_objs[1].tick_params(axis="both", labelsize=8)
    ax_objs[1].grid(True)

    ax_objs[0].set_xlim(poi["lon"].min(), poi["lon"].max())
    ax_objs[0].set_ylim(poi["lat"].min(), poi["lat"].max())
    ax_objs[1].set_xlim(poi["lon"].min(), poi["lon"].max())
    ax_objs[1].set_ylim(poi["lat"].min(), poi["lat"].max())
    plt.tight_layout()


def load_concat_train_test_ais():
    """Merging the training and testing data."""
    train_data = load_pkl_from_path("train_semantic_tmp.pkl")
    test_data = load_pkl_from_path("test_semantic_tmp.pkl")
    ais_data = load_pkl_from_path("ais_semantic_tmp.pkl")
    concat_data = train_data + test_data + ais_data
    return concat_data, len(train_data), len(test_data)


def label_traj_data_semantics(traj_data=None, nn=None, pois=None):
    """Labeling each GPS points with semantic label."""
    pois = pois.reset_index().rename({"index":"is_stop"}, axis=1)
    traj_new_data = []
    for ind, traj in tqdm(enumerate(traj_data), total=len(traj_data)):
        nn_dist, nn_ind = nn.radius_neighbors(traj[["x", "y"]].values,
                                              return_distance=True)
        nn_ind = [list(i) for i in nn_ind]
        traj["is_stop"] =  list(map(lambda x: x[0] if len(x) != 0 else -1, nn_ind))
        traj = pd.merge(traj, pois, how="left", on="is_stop")
        traj_new_data.append(traj)
    return traj_new_data


@timefn
def concat_list_data(data_list=None, local_file_name=None):
    """Concating all trajectories in data_list."""
    PATH=".//tcdata_tmp//"
    file_processor = LoadSave()
    data_concat = pd.concat(data_list, axis=0, ignore_index=True)
    file_processor.save_data(path=PATH + local_file_name,
                             data=data_concat)


def poi_classification():
    """Two types of POIs: 1. In port, 2: On the sea."""
    pois = pd.read_csv(".//tcdata_tmp//pois_baidu_query.csv", encoding="GBK")
    clf = DBSCAN(eps=800, min_samples=100, n_jobs=-1, algorithm="kd_tree")
    pois["label"] = clf.fit_predict(pois[["x", "y"]].values,
        sample_weight=pois["visit_count"].values)
    pois = pois.dropna(how="any", axis=1)

    basic_json = []
    for i in range(len(pois)):
        tmp = pois["result"].iloc[i]
        tmp = json.loads(tmp)
        basic_json.append(tmp["result"])

    address_info = []
    for i in range(len(basic_json)):
        tmp = basic_json[i]["addressComponent"]
        address_info.append(tmp)
    address_info = pd.DataFrame(address_info)
    pois = pd.concat([pois, address_info], axis=1)

    pois["possible_port"] = pois["country_code"] != -1
    # districe encoding
    district_dict = {}
    for ind, name in enumerate(pois["district"].unique()):
        if name != "":
            district_dict[name] = ind
        else:
            district_dict[name] = np.nan
    pois["district"].replace(district_dict, inplace=True)

    # city encoding
    city_dict = {}
    for ind, name in enumerate(pois["city"].unique()):
        if name != "":
            city_dict[name] = ind
        else:
            city_dict[name] = np.nan
    pois["city"].replace(city_dict, inplace=True)
    pois = pois[["label", "x", "y", "city", "district", "possible_port"]]
    return pois


def traj_data_labeling_semantics():
    '''
    Step 1: Load all possible stop grids.
    '''
    traj_data_list, train_nums, test_nums = load_concat_train_test_ais()
    pois = poi_classification()

    '''
    Step 2: Find all candiate stop points.
    '''
    # Label all semantic points
    nn = NearestNeighbors(n_neighbors=1, radius=400)
    clf = nn.fit(pois[["x", "y"]].values)
    traj_data_semantic = label_traj_data_semantics(
        traj_data_list, clf, pois.drop(["x", "y"], axis=1))

    # Spliting the training and testing data
    train_data = traj_data_semantic[:train_nums]
    test_data = traj_data_semantic[train_nums:(train_nums+test_nums)]
    ais_data = traj_data_semantic[(train_nums+test_nums):]

    # Save all data and concat the training and testing data
    print("\n@Semantic labeling results:")
    print("-----------------------------")
    print("#training: {}, #testing: {}, #AIS: {}.".format(
          len(train_data), len(test_data), len(ais_data)))
    print("-----------------------------\n")

    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//train_semantic.pkl",
                             data=train_data)
    file_processor.save_data(path=".//tcdata_tmp//test_semantic.pkl",
                             data=test_data)
    file_processor.save_data(path=".//tcdata_tmp//ais_semantic.pkl",
                             data=ais_data)

    '''
    Step 3: Concat a list of traj data.
    '''
    concat_list_data(data_list=train_data,
                     local_file_name="train_semantic_concat.pkl")
    concat_list_data(data_list=test_data,
                     local_file_name="test_semantic_concat.pkl")


if __name__ == "__main__":
    traj_data_labeling_semantics()
    pois = poi_classification()
    