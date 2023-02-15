import numpy as np
import pandas as pd
import time
import h5pyd
import geopandas as gpd
from dw_tap.data_fetching import getData
from dw_tap.power_output import estimate_power_output
from dw_tap.lom import run_lom

z_turbine = 40 # turbine height in [m]
lat, lon = 39.32856, -89.40238
obstacle_file = "../sites/180-Manual1.geojson"

t_start = time.time()

#Read in data from CSV file instead of hsds + getData
atmospheric_df = pd.read_csv("../data/180_1year_12hourgranularity.csv")
# Temporary (later need to resave csv without index column) 
if "Unnamed: 0" in atmospheric_df.columns:
    atmospheric_df.drop(columns=["Unnamed: 0"], inplace=True)

obstacles_df = gpd.read_file(obstacle_file)
# Leave in only relevant columns
obstacles_df = obstacles_df[["height", "geometry"]]

x1_turbine, y1_turbine = lat, lon
xy_turbine = [np.array([x1_turbine, y1_turbine])]

t_lom_start = time.time()
predictions_df = \
    run_lom(atmospheric_df, obstacles_df, xy_turbine, z_turbine)
t_lom = time.time() - t_lom_start
print('Running LOM: %.2f (s)' % t_lom)

t_power_start = time.time()
kw, above_curve, below_curve, above_curve_list, below_curve_list = \
    estimate_power_output(predictions_df, atmospheric_df["temp"], atmospheric_df["pres"])  
t_power = time.time() - t_power_start
print('Power estimation: %.2f (s)' % t_power)

t_total = time.time()-t_start
print('Total time: %.2f (s)' % t_total)

