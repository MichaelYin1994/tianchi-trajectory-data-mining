#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:20:19 2020

@author: zhuoyin94
"""

import warnings
import multiprocessing as mp
from functools import partial
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def load_fishing_ground():
    file_processor = LoadSave()
    data = file_processor.load_data(".//tcdata//fishing_ground.pkl")
    return data


def pnpoly(poly_vert_list=None, test_point=None):
    """Which polygon the test_point belongs to ?

    Each element in the poly_vert_list is a polygon with list type.
    """
    for i, polygon in enumerate(poly_vert_list):
        vert_count = len(polygon)
        is_inside = False
        ii, jj = 0, vert_count - 1

        while(ii < vert_count):
            if (polygon[ii][1] > test_point[1]) != (polygon[jj][1] > test_point[1]):
                if test_point[0] < ((polygon[jj][0] - polygon[ii][0]) * (test_point[1] - polygon[ii][1]) / (polygon[jj][1] - polygon[ii][1]) + polygon[ii][0]):
                    is_inside = not is_inside
            jj = ii
            ii += 1

        if is_inside:
            return i
    return np.nan


def find_fishing_ground(traj=None, poly_vert_list=None):
    test_point_list = traj[["lon", "lat"]].values.tolist()

    fishing_ground_ind = []
    for test_point in test_point_list:
        fishing_ground_ind.append(pnpoly(poly_vert_list, test_point))
    traj["fishing_ground"] = fishing_ground_ind
    return traj


def traj_to_bin(traj=None, x_min=12031967.16239096, x_max=14226964.881853,
                y_min=1623579.449434373, y_max=4689471.1780792,
                row_bins=4380, col_bins=3136):
    # col_bins = (14226964.881853 - 12031967.16239096) / 700
    # row_bins = (4689471.1780792 - 1623579.449434373) / 3000
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
    traj.sort_values(by='time', inplace=True)
    return traj


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
    ax_objs[0].scatter(poi_filtered["lon"].values, poi_filtered["lat"].values, s=9,
        alpha=1, marker=".", color="red", label="REAL")
    ax_objs[0].tick_params(axis="both", labelsize=8)
    ax_objs[0].grid(True)
    ax_objs[0].legend(fontsize=12)

    ax_objs[1].scatter(poi_filtered["lon"].values, poi_filtered["lat"].values, s=9,
        alpha=1, marker=".", color="red")
    ax_objs[1].tick_params(axis="both", labelsize=8)
    ax_objs[1].grid(True)

    ax_objs[0].set_xlim(poi["lon"].min(), poi["lon"].max())
    ax_objs[0].set_ylim(poi["lat"].min(), poi["lat"].max())
    ax_objs[1].set_xlim(poi["lon"].min(), poi["lon"].max())
    ax_objs[1].set_ylim(poi["lat"].min(), poi["lat"].max())
    plt.tight_layout()


def load_concat_train_test_ais():
    """Merging the training and testing data."""
    train_data = load_pkl_from_path("train.pkl")
    test_data = load_pkl_from_path("test.pkl")
    ais_data = load_pkl_from_path("ais.pkl")
    concat_data = train_data + test_data + ais_data
    return concat_data, len(train_data), len(test_data)


def traj_data_semantic_label(traj_data=None, nn=None):
    """Labeling each GPS points with semantic label."""
    for ind, traj in enumerate(traj_data):
        nn_dist, nn_ind = nn.radius_neighbors(traj[["x", "y"]].values,
                                              return_distance=True)
        nn_ind = [list(i) for i in nn_ind]
        traj["is_stop"] =  list(map(lambda x: x[0] if len(x) != 0 else -1, nn_ind))
    return traj_data


@timefn
def concat_list_data(data_list=None, local_file_name=None):
    """Concating all trajectories in data_list."""
    PATH=".//tcdata_tmp//"
    file_processor = LoadSave()
    data_concat = pd.concat(data_list, axis=0, ignore_index=True)
    file_processor.save_data(path=PATH + local_file_name,
                             data=data_concat)


def find_save_visit_count_table(traj_data_df=None, bin_to_coord_df=None):
    """Find and save the visit frequency of each bin."""
    visit_count_df = traj_data_df.groupby(["no_bin"]).count().reset_index()
    visit_count_df = visit_count_df[["no_bin", "x"]]
    visit_count_df.rename({"x":"visit_count"}, axis=1, inplace=True)

    visit_count_df_save = pd.merge(bin_to_coord_df, visit_count_df, on="no_bin", how="left")
    file_processor = LoadSave()
    file_processor.save_data(data=visit_count_df_save,
                             path=".//tcdata_tmp//bin_visit_count_frequency.pkl")
    return visit_count_df


def find_save_unique_visit_count_table(traj_data_df=None, bin_to_coord_df=None):
    """Find and save the unique boat visit count of each bin."""
    unique_boat_count_df = traj_data_df.groupby(["no_bin"])["boat_id"].nunique().reset_index()
    unique_boat_count_df.rename({"boat_id":"visit_boat_count"}, axis=1, inplace=True)

    unique_boat_count_df_save = pd.merge(bin_to_coord_df, unique_boat_count_df,
                                         on="no_bin", how="left")
    file_processor = LoadSave()
    file_processor.save_data(data=unique_boat_count_df_save,
                             path=".//tcdata_tmp//bin_unique_boat_count_frequency.pkl")
    return unique_boat_count_df


def find_save_mean_stay_time_table(traj_data_df=None, bin_to_coord_df=None):
    """Find and save the mean stay time of each bin."""
    mean_stay_time_df = traj_data_df.groupby(
        ["no_bin", "boat_id"])["time_array"].sum().reset_index()
    mean_stay_time_df.rename({"time_array":"total_stay_time"}, axis=1, inplace=True)
    mean_stay_time_df = mean_stay_time_df.groupby(
        ["no_bin"])["total_stay_time"].mean().reset_index()
    mean_stay_time_df.rename(
        {"total_stay_time":"mean_stay_time"}, axis=1, inplace=True)

    mean_stay_time_df_save = pd.merge(bin_to_coord_df, mean_stay_time_df,
                                      on="no_bin", how="left")
    file_processor = LoadSave()
    file_processor.save_data(data=mean_stay_time_df_save,
                             path=".//tcdata_tmp//bin_mean_stay_time.pkl")
    return mean_stay_time_df


def traj_data_poi_mining(visit_count_minimum=200, visit_boat_minimum=3,
                         mean_stay_minutes=120, bin_size=800):
    '''
    Step 1: Find all possible stop grids.
    '''
    traj_data_list, train_nums, test_nums = load_concat_train_test_ais()

    print("\n@Step 1: traj2bin:")
    print("-----------------------------")
    col_bins = int((14226964.881853 - 12031967.16239096) / bin_size)
    row_bins = int((4689471.1780792 - 1623579.449434373) / bin_size)
    partial_work = partial(traj_to_bin, col_bins=col_bins, row_bins=row_bins)
    with mp.Pool(processes=mp.cpu_count()) as p:
        res = list(tqdm(p.imap(partial_work, traj_data_list),
                        total=len(traj_data_list)))
    print("-----------------------------")

    traj_data_df = [traj[["x", "y", "no_bin", "lon",
                          "lat", "boat_id", "time_array"]] for traj in res]
    traj_data_df = pd.concat(traj_data_df, axis=0, ignore_index=True)
    bin_to_coord_df = traj_data_df.groupby(
        ["no_bin"]).median().reset_index().drop(["boat_id"], axis=1)

    # DataFrame tmp for finding POIs
    visit_count_df = find_save_visit_count_table(
        traj_data_df, bin_to_coord_df)
    unique_boat_count_df = find_save_unique_visit_count_table(
        traj_data_df, bin_to_coord_df)
    mean_stay_time_df = find_save_mean_stay_time_table(
        traj_data_df, bin_to_coord_df)

    candidate_pois = visit_count_df.query(
        "visit_count >= {}".format(visit_count_minimum)).reset_index(drop=True)

    candidate_pois = pd.merge(
        candidate_pois, unique_boat_count_df, on="no_bin", how="left")
    candidate_pois = candidate_pois.query(
        "visit_boat_count >=  {}".format(visit_boat_minimum)).reset_index(drop=True)

    candidate_pois = pd.merge(
        candidate_pois, mean_stay_time_df, on="no_bin", how="left")
    candidate_pois = candidate_pois.query(
        "mean_stay_time >=  {}".format(mean_stay_minutes)).reset_index(drop=True)

    candidate_pois = pd.merge(
        candidate_pois, bin_to_coord_df, on="no_bin", how="left")
    candidate_pois.drop(["time_array"], axis=1, inplace=True)

    clf = DBSCAN(eps=1500, min_samples=200, n_jobs=-1, algorithm="kd_tree")
    candidate_pois["label"] = clf.fit_predict(candidate_pois[["x", "y"]].values,
        sample_weight=candidate_pois["visit_count"].values)
    pois = candidate_pois[candidate_pois["label"] != -1]
    pois.to_csv(".//tcdata_tmp//pois.csv", index=False)

    # Labeling fishing ground
    fishing_ground = load_fishing_ground()
    fishing_ground_polygons = fishing_ground["arr"].values.tolist()
    print("\n********************")
    print("@AIS preprocessing start at: {}".format(datetime.now()))
    print("********************")
    partial_work = partial(find_fishing_ground, poly_vert_list=fishing_ground_polygons)
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(partial_work, traj_data_list),
                        total=len(traj_data_list)))
    print("\n********************")
    print("@AIS preprocessing ended at: {}".format(datetime.now()))
    print("********************")
    traj_data_semantic = tmp

    # Spliting the training and testing data
    train_data = traj_data_semantic[:train_nums]
    test_data = traj_data_semantic[train_nums:(train_nums+test_nums)]
    ais_data = traj_data_semantic[(train_nums+test_nums):]

    # Save all data and concat the training and testing data
    print("\n@Semantic labeling results:")
    print("-----------------------------")
    print("#training: {}, #testing A: {}, #AIS: {}.".format(
          len(train_data), len(test_data), len(ais_data)))
    print("-----------------------------\n")

    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//train_semantic_tmp.pkl",
                             data=train_data)
    file_processor.save_data(path=".//tcdata_tmp//test_semantic_tmp.pkl",
                             data=test_data)
    file_processor.save_data(path=".//tcdata_tmp//ais_semantic_tmp.pkl",
                             data=ais_data)
    file_processor.save_data(path=".//tcdata_tmp//pois.pkl",
                             data=pois)

    return candidate_pois, pois


if __name__ == "__main__":
    candidate_pois, pois = traj_data_poi_mining()

    plot = True
    if plot:
        plot_poi_filtered_poi_coordinates(candidate_pois, pois)
