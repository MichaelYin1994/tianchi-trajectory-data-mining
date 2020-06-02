#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:54:10 2020

@author: zhuoyin94
"""

import os
import multiprocessing as mp
import warnings
import gc
from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import seaborn as sns
from utils import timefn, haversine_np, LoadSave

np.random.seed(2019)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def read_preprocess_csv_from_path(nrows=100, nfiles=1, sampling_freq="10min"):
    PATH = ".//tcdata//round2_ais_20200310//"
    ais_csv_names = sorted(os.listdir(PATH))
    ais_csv_count = len(ais_csv_names)

    if nfiles is None:
        nfiles = ais_csv_count
    elif nfiles <= 0 or nfiles >= ais_csv_count:
        nfiles = 1

    ais_csv_list = []
    print("\n@Load and SAMPLE the raw AIS csv:")
    print("---------------------------------------")
    for ind, ais_csv_name in enumerate(ais_csv_names[:nfiles]):
        df = pd.read_csv(PATH + ais_csv_name, nrows=nrows)
        df = preprocessing_raw_ais(df, sampling_freq=sampling_freq)
        ais_csv_list.append(df)
        print("--Now index at {}, completed {:.2f}%.".format(ind,
            (ind + 1) / nfiles * 100))

    ais_csv = pd.concat(ais_csv_list, axis=0, ignore_index=True)
    ais_csv.sort_values(by=["ais_id", "time"],
                        ascending=[True, True], inplace=True)
    ais_csv.reset_index(drop=True, inplace=True)
    print("---------------------------------------")
    return ais_csv


def reformat_strtime(time_str=None, START_YEAR="2019"):
    """Reformat the strtime with the form '08 14' to 'START_YEAR-08-14' """
    time_str_split = time_str.split(" ")
    time_str_reformat = START_YEAR + "-" + time_str_split[0][:2] + "-" + time_str_split[0][2:4]
    time_str_reformat = time_str_reformat + " " + time_str_split[1]
    return time_str_reformat


def preprocessing_raw_ais(ais=None, sampling_freq="10min"):
    # Renaming the columns
    ais.rename({"船速": "speed", "航向": "direction"}, axis=1, inplace=True)

    # Reformat the time feature
    ais["time"] = ais["time"].apply(reformat_strtime)
    ais["time"] = pd.to_datetime(ais["time"])
    ais.set_index(ais["time"], drop=False, inplace=True)

    # GROUPBY AND RESAMPLE
    ais = ais.groupby(["ais_id"]).resample(sampling_freq,
                     label="left").median().dropna(how="all")
    ais = ais.drop(["ais_id"], axis=1).reset_index()
    return ais


################################ Preprocessing ################################
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
                                   time_interval_maximum=200,
                                   coord_speed_maximum=700):
    """Assign the anomaly points in traj to np.nan."""
    # Step 1: The speed anomaly repairing
    is_speed_anomaly = (traj["speed"] > speed_maximum) | (traj["speed"] < 0)
    traj["speed"][is_speed_anomaly] = np.nan
    speed_change_count = np.sum(is_speed_anomaly)

    # Step 2: Anomaly coordinates repairing
    is_anomaly = np.array([False] * len(traj))
    traj["coord_speed"] = traj["dist_array"] / traj["time_array"]
    
    # Condition 1: huge coord speed and huge time interval
    is_anomaly_tmp = (traj["time_array"] > time_interval_maximum) | (traj["coord_speed"] > coord_speed_maximum)
    is_anomaly = is_anomaly | is_anomaly_tmp

    # Condition 2: coordinate not in CHINA
    is_anomaly_lon_tmp = (traj["lon"] < 106.38) | (traj["lon"] > 132)
    is_anomaly_lat_tmp = (traj["lat"] < 3.836) | (traj["lat"] > 42.9)
    is_anomaly = is_anomaly | is_anomaly_lon_tmp | is_anomaly_lat_tmp

    # Condition 3: 3-\sigma method
    traj = traj[~is_anomaly].reset_index(drop=True)
    is_anomaly = np.array([False] * len(traj))

    if len(traj) != 0:
        lon_std, lon_mean = traj["lon"].std(), traj["lon"].mean()
        lat_std, lat_mean = traj["lat"].std(), traj["lat"].mean()
        lon_low, lon_high = lon_mean - 3 * lon_std, lon_mean + 3 * lon_std
        lat_low, lat_high = lat_mean - 3 * lat_std, lat_mean + 3 * lat_std

        is_anomaly = is_anomaly | (traj["lon"] > lon_high) | ((traj["lon"] < lon_low))
        is_anomaly = is_anomaly | (traj["lat"] > lat_high) | ((traj["lat"] < lat_low))
        traj = traj[~is_anomaly].reset_index(drop=True)
    return traj, [speed_change_count, len(is_speed_anomaly) - len(traj)]


def preprocessing_traj(traj=None):
    """Preprocessing a single trajectory."""
    traj.reset_index(drop=True, inplace=True)

    # Coordonate projection(From WGS-84(EPSG:4326) to EPSG:3395 Mercator)
    traj["x"], traj["y"] = traj["lon"].copy(), traj["lat"].copy()
    traj_geo = [Point(xy) for xy in zip(traj.x, traj.y)]
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(traj_geo, crs=crs, geometry=traj_geo)
    gdf = gdf.to_crs(crs="epsg:3395")
    traj["x"], traj["y"] = gdf["geometry"].x, gdf["geometry"].y

    # Reformatting str time
    traj = compute_traj_diff_time_distance(traj)

    # Anomaly points interpolate
    traj, c_rec = assign_traj_anomaly_points_nan(traj.copy())
    traj["speed"] = traj["speed"].interpolate(method="linear", axis=0)
    traj = traj.fillna(method="bfill")
    traj = traj.fillna(method="ffill")
    traj["speed"] = traj["speed"].clip(0, 23)
    traj.rename({"ais_id":"boat_id"}, axis=1, inplace=True)
    traj["boat_id"] = traj["boat_id"] + 10000000
    return traj, c_rec


def save_ais_traj_to_csv(ais=None, round_to_print=50000,
                         local_file_name="ais.pkl"):
    """Save the trajectory according to the ais record with the csv format."""
    ais_id_list = ais["ais_id"].astype(int).values.tolist()

    # Split the DataFrame
    ais_traj_list = []
    head, tail = 0, 0
    print("\n@Split AIS and save the traj in *.csv format:")
    print("---------------------------------------")
    while(tail <= (len(ais_id_list) - 1)):
        if tail % round_to_print == 0:
            print("--Now tail is on {}, completed {:.2f}%.".format(tail,
                (tail + 1) / len(ais_id_list) * 100))
            print("--time is {}.\n".format(datetime.now()))
        if ais_id_list[head] == ais_id_list[tail]:
            tail += 1
        elif ais_id_list[head] != ais_id_list[tail]:
            ais_traj_list.append(ais.iloc[head:tail])
            head = tail
    ais_traj_list.append(ais.iloc[head:])
    print("---------------------------------------")

    # Coordinate transferring
#    tmp = []
#    for i in range(50):
#        tmp.append(preprocessing_traj(ais_traj_list[i]))
    print("\n@AIS list index resetting:")
    print("---------------------------------------")
    with mp.Pool(processes=mp.cpu_count()) as p:
        tmp = list(tqdm(p.imap(preprocessing_traj, ais_traj_list),
                        total=len(ais_traj_list)))
    print("---------------------------------------")
    print("@Save to the local file: {}.\n".format(
        local_file_name))
    traj_data = [item[0] for item in tmp if len(item[0]) > 1]
    change_record = [item[1] for item in tmp]
    change_record = pd.DataFrame(change_record,
                                 columns=["speed_change", "coord_change"])

    # Saving processed data to the lcoal path with *.pkl format
    file_processor = LoadSave()
    file_processor.save_data(path=".//tcdata_tmp//{}".format(local_file_name),
                             data=traj_data)
    return traj_data, change_record


def ais_data_processing():
    print("\n********************")
    print("@AIS preprocessing start at: {}".format(datetime.now()))
    print("********************")

    # Step 1: Read and preprocess the raw csv
    ais = read_preprocess_csv_from_path(nrows=None, nfiles=None)

    # Step 2: Noisy points filtering and coordinates system mapping.
    traj_data, traj_change_record = save_ais_traj_to_csv(ais, round_to_print=1000000)
    gc.collect()

    print("\n********************")
    print("@AIS preprocessing ended at: {}".format(datetime.now()))
    print("********************")    


if __name__ == "__main__":
    print("\n********************")
    print("@AIS preprocessing start at: {}".format(datetime.now()))
    print("********************")

    # Step 1: Read and preprocess the raw csv
    ais = read_preprocess_csv_from_path(nrows=None, nfiles=None)

    # Step 2: Noisy points filtering and coordinates system mapping.
    traj_data, traj_change_record = save_ais_traj_to_csv(ais, round_to_print=1000000)
    gc.collect()

    print("\n********************")
    print("@AIS preprocessing ended at: {}".format(datetime.now()))
    print("********************")
