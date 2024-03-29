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
import time
import warnings
import logging
logger = logging.getLogger()

# Hide all warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.append("../scripts")
import dw_tap_data 

def test_pilowf_p1z2():
    index = pd.read_csv(os.path.join(dw_tap_data.path, "notebook_data/01_one_energy_turbine_data/OneEnergyTurbineData.csv"))
    
    tid = 'p1z2'
    #wind_source = "wtk_led_2018"
    obstacle_mode = "treesasbldgs_100m"

    pid = tid[:2]
    row = index[index["APRS ID"] == tid].iloc[0]
    lat = row["Latitude"]
    lon = row["Longitude"]
    z_turbine = row["Hub Height (m)"]
    xy_turbine = [np.array([lon, lat])]
    
    # Prepare a small wind input for this case
    
    start = pd.to_datetime('2018-04-25 06:30:00+00:00') 
    end = pd.to_datetime('2018-04-25 08:55:00+00:00')
    
    wtk_led_2018 = pd.read_csv(os.path.join(dw_tap_data.path, "notebook_data/01_one_energy_turbine_data/wtk_led_2018.csv.bz2"))
    
    tmp = wtk_led_2018[wtk_led_2018["tid"] == tid].copy().reset_index(drop=True)
    tmp["datetime"] = pd.to_datetime(tmp["packet_date"])
    tmp = tmp[(tmp["datetime"] >= start) & (tmp["datetime"] <= end)].reset_index(drop=True)
    
    atmospheric_input = tmp
    
    # Prepare obstacles input
    
    obs = AllObstacles(data_dir=os.path.join(dw_tap_data.path, "notebook_data"), types=["oneenergy"], debug=False)
    obs_subset = obs.get("oneenergy", tid, obstacle_mode)
    logger.info("Fetched obstacles. Total # of obstacles: %.d" % len(obs_subset))
    
    # Run ANL's LOM
    t_lom_start = time.time() 
    predictions_df = run_lom(atmospheric_input, obs_subset, xy_turbine, z_turbine, check_distance=True)
    t_lom = time.time() - t_lom_start    
    logger.info("Reporting the runtime for PILOWF: %.3f (s)" % t_lom) 

    pmin = predictions_df["ws-adjusted"].min() 
    pavg = predictions_df["ws-adjusted"].mean() 
    pmax = predictions_df["ws-adjusted"].max() 

    logger.info("PILOWF's ws-adjusted, average (%d timesteps): %.3f" % (len(predictions_df), pavg))

    assert pmin >= -10, "Testing PILOWF: realistic min ws-adjusted should be >=-10 m/s (observed: %f)." % pmin
    assert pavg >= 0, "Testing PILOWF: avg ws-adjusted should be non-negative (observed: %f)." % pavg
    assert pmax <= 150, "Testing PILOWF: realistic max ws-adjusted should be <= 150 m/s (observed: %f)." % pmax
