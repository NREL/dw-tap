import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
from dw_tap.lom import run_lom
import os
import seaborn as sns
import glob
from dw_tap.data_processing import _LatLon_To_XY, filter_obstacles
import subprocess
import shutil
from dw_tap.obstacles import AllObstacles
from dw_tap.lom import run_lom

import sys
sys.path.append("../scripts")
import dw_tap_data 

def test_check_for_negative_ws_p1z2():
    index = pd.read_csv(os.path.join(dw_tap_data.path, "01 One Energy Turbine Data/OneEnergyTurbineData.csv"))
    
    tid = 'p1z2'
    #wind_source = "wtk_led_2018"
    obstacle_mode = "treesasbldgs_100m"
    
    # Prepare a small wind input for this case
    
    start = pd.to_datetime('2018-04-25 06:30:00+00:00') 
    end = pd.to_datetime('2018-04-25 08:55:00+00:00')
    
    wtk_led_2018 = pd.read_csv(os.path.join(dw_tap_data.path, "01 One Energy Turbine Data/wtk_led_2018.csv.bz2"))
    
    tmp = wtk_led_2018[wtk_led_2018["tid"] == tid].copy().reset_index(drop=True)
    tmp["datetime"] = pd.to_datetime(tmp["packet_date"])
    tmp = tmp[(tmp["datetime"] >= start) & (tmp["datetime"] <= end)].reset_index(drop=True)
    
    atmospheric_input = tmp
    
    # Prepare obstacles input
    
    obs = AllObstacles(data_dir=dw_tap_data.path, types=["oneenergy"], debug=False)
    obs_subset = obs.get("oneenergy", tid, obstacle_mode)
    #obs_subset
    
    # Run ANL's LOM
    
    pid = tid[:2]
    row = index[index["APRS ID"] == tid].iloc[0]
    lat = row["Latitude"]
    lon = row["Longitude"]
    z_turbine = row["Hub Height (m)"]
    xy_turbine = [np.array([lon, lat])]
    print(lat, lon, z_turbine)
    
    predictions_df = run_lom(atmospheric_input, obs_subset, xy_turbine, z_turbine, check_distance=True)
    #predictions_df
 
    pmin = predictions_df["ws-adjusted"].min() 
    pmavg = predictions_df["ws-adjusted"].mean() 
    assert pmin > 0, "Testing PILOWF: min ws-adjusted should be non-negative (observed: %f)." % pmin
    assert pavg > 0, "Testing PILOWF: avg ws-adjusted should be non-negative (observed: %f)." % pavg
