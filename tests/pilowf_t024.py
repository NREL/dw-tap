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
import warnings
import time
import logging
logger = logging.getLogger()

# Hide all warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.append("../scripts")
import dw_tap_data 

def test_pilowf_t024():
    index = pd.read_csv(os.path.join(dw_tap_data.path, "notebook_data/01_bergey_turbine_data/bergey_sites.csv"))
    
    tid = 't024'
    #wind_source = "wtk_led_2018"
    obstacle_mode = "treesasbldgs"

    row = index[index["APRS ID"] == tid].iloc[0]
    lat = row["Latitude"]
    lon = row["Longitude"]
    z_turbine = row["Hub Height (m)"]
    xy_turbine = [np.array([lon, lat])]
    
    wtk = pd.read_csv(os.path.join(dw_tap_data.path, "notebook_data/01_bergey_turbine_data/wtk_tp.csv.bz2"))
    
    tmp = wtk[wtk["tid"] == tid].copy().reset_index(drop=True)
    tmp["datetime"] = pd.to_datetime(tmp["packet_date"])
       
    # 1 year of data
    start = pd.to_datetime('2013-01-01 00:00:00+00:00') 
    end = pd.to_datetime('2013-12-31 23:59:00+00:00')
    tmp = tmp[(tmp["datetime"] >= start) & (tmp["datetime"] <= end)].reset_index(drop=True)
    
    atmospheric_input = tmp
    
    #print("atmospheric_input:")
    #display(atmospheric_input)
    
    # Prepare obstacles input
    
    obs = AllObstacles(data_dir=os.path.join(dw_tap_data.path, "notebook_data"), types=["bergey"], debug=False)
    obs_subset = obs.get("bergey", tid, obstacle_mode)
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
