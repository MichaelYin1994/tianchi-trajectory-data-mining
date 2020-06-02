#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:41:41 2020

@author: yinzhuo
"""
import os
import warnings
import multiprocessing as mp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import seaborn as sns
from tqdm import tqdm
from utils import LoadSave, haversine_np

np.random.seed(2019)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def reformat_strtime(time_str=None, START_YEAR="2019"):
    """Reformat the str-time of the form '08 14' to 'START_YEAR-08-14' """
    splits = time_str.split(" ")
    time_str_reformat = START_YEAR + "-" + splits[0][:2] + "-" + splits[0][2:4]
    time_str_reformat = time_str_reformat + " " + splits[1]
    return time_str_reformat


def compute_traj_diff_time_distance(traj=None):
    """Compute the sampling time and the coordinate distance."""
    # Compute the time difference
    time_diff_array = (traj["time"].iloc[1:].reset_index(drop=True) - traj[
        "time"].iloc[:-1].reset_index(drop=True)).dt.total_seconds() / 60
#    dist_array = np.sqrt(np.diff(traj[x].values)**2 + np.diff(traj[y].values)**2)

    # Compute the coordinate distance
    dist_diff_array = haversine_np(traj["lon"].values[1:],  # lon_0
                                   traj["lat"].values[1:],  # lat_0
                                   traj["lon"].values[:-1], # lon_1
                                   traj["lat"].values[:-1]  # lat_1
                                   )

    # Filling the fake values
    time_diff_array = [0.5] + time_diff_array.tolist()
    dist_diff_array = [0.5] + dist_diff_array.tolist()
    traj["time_array"] = time_diff_array
    traj["dist_array"] = dist_diff_array
    return traj


def assign_traj_anomaly_points_nan(traj=None, speed_maximum=23,
                                   coord_speed_window_size=2,
                                   coord_speed_maximum=700):
    """Assign the anomaly points in traj to np.nan."""
    # Step 1: The speed anomaly repairing
    is_speed_anomaly = (traj["speed"] > speed_maximum) | (traj["speed"] < 0)
    traj["speed"][is_speed_anomaly] = np.nan
    speed_change_count = np.sum(is_speed_anomaly)

    # Step 2: Anomaly coordinates repairing
    if ((np.sum(traj["lon"].values <= 1)) != 0) or ((np.sum(traj["lat"].values <= 1)) != 0):
        cond = (traj["lon"].values <= 1) | (traj["lat"].values <= 1)
        traj["x"][cond] = np.nan
        traj["y"][cond] = np.nan
        traj["lon"][cond] = np.nan
        traj["lat"][cond] = np.nan
        return traj, [speed_change_count, sum(cond)]

    # Compute the average speed of each specified coordinates
    dist_rolling = traj["dist_array"].rolling(
        window=coord_speed_window_size).sum()
    time_rolling = traj["time_array"].rolling(
        window=coord_speed_window_size).sum()
    mean_coord_speed = dist_rolling / time_rolling
    possible_noise_index = np.arange(0, len(traj))
    possible_noise_index = possible_noise_index[
        mean_coord_speed >= coord_speed_maximum]

    x_original, y_original = traj["x"].values.copy(), traj["y"].values.copy()
    lon_original, lat_original = traj["lon"].values.copy(), traj["lat"].values.copy()
    for ind in possible_noise_index:
        x_original[ind] = np.nan
        y_original[ind] = np.nan
        lon_original[ind] = np.nan
        lat_original[ind] = np.nan
    coord_change_count = len(possible_noise_index)
    traj["x"], traj["y"] = x_original, y_original
    traj["lon"], traj["lat"] = lon_original, lat_original
    return traj, [speed_change_count, coord_change_count]


def preprocessing_traj(traj=None):
    """Preprocessing a single trajectory."""
    traj.rename({"渔船ID": "boat_id",
                 "速度": "speed",
                 "方向": "direction"}, axis=1, inplace=True)
    if "type" in traj.columns:
        traj["type"].replace({"刺网": 2, "围网": 1, "拖网": 0}, inplace=True)

    # Coordonate projection(From WGS-84(EPSG:4326) to EPSG:3395 Mercator)
    traj["x"], traj["y"] = traj["lon"].copy(), traj["lat"].copy()
    traj_geo = [Point(xy) for xy in zip(traj.x, traj.y)]
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(traj_geo, crs=crs, geometry=traj_geo)
    gdf = gdf.to_crs(crs="epsg:3395")
    traj["x"], traj["y"] = gdf["geometry"].x, gdf["geometry"].y

    # Reformatting str time
    traj["time"] = traj["time"].apply(reformat_strtime)
    traj["time"] = pd.to_datetime(traj["time"])
    traj = traj.reindex(index=traj.index[::-1]).reset_index(drop=True)
    traj = compute_traj_diff_time_distance(traj)

    # Anomaly points interpolate
    traj, c_rec = assign_traj_anomaly_points_nan(traj.copy())
    traj = traj.interpolate(method="polynomial", axis=0, order=2)
    traj = traj.fillna(method="bfill")
    traj = traj.fillna(method="ffill")
    traj["speed"] = traj["speed"].clip(0, 23)
    return traj, c_rec


def preprocessing_raw_csv(PATH=".//tcdata//hy_round2_train_20200225//",
                          local_file_name="train.pkl"):
    """Loading and processing all train csv data."""
    if PATH is None:
        raise ValueError("Invalid PATH !")
    file_names = sorted(os.listdir(PATH), key=lambda s: int(s.split(".")[0]))

    # Loading all trajectory data.
    traj_data = []
    for name in file_names:
        traj_data.append(pd.read_csv(PATH + name, encoding="utf-8"))

    # Processing each trajectory data.
    print("\n@Multi-processing RAW CSV started:")
    print("-----------------------------")
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(preprocessing_traj, traj_data),
                        total=len(traj_data)))
    print("-----------------------------")
    print("@Multi-processing RAW CSV ended, to the local file: {}.\n".format(
        local_file_name))
    traj_data = [item[0] for item in tmp]
    change_record = [item[1] for item in tmp]
    change_record = pd.DataFrame(change_record,
                                 columns=["speed_change", "coord_change"])

    # Saving processed data to the lcoal path with *.pkl format
    file_processor = LoadSave(PATH)
    file_processor.save_data(path=".//tcdata_tmp//{}".format(local_file_name),
                             data=traj_data)
    return change_record


def traj_data_preprocessing_csv():
    _ = preprocessing_raw_csv(PATH=".//tcdata//hy_round2_train_20200225//",
                              local_file_name="train.pkl")
    _ = preprocessing_raw_csv(PATH=".//tcdata//hy_round2_testA_20200225//",
                              local_file_name="test.pkl")


if __name__ == "__main__":
    c_rec_train = preprocessing_raw_csv(PATH=".//tcdata//hy_round2_train_20200225//",
                                        local_file_name="train.pkl")
    c_rec_test = preprocessing_raw_csv(PATH=".//tcdata//hy_round2_testA_20200225//",
                                       local_file_name="test.pkl")
