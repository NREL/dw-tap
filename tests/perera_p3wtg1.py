# The following allows finding data directory based on the config in ~/.tap.ini
import sys
sys.path.append("../scripts")
import dw_tap_data 
# After 3 lines a above data directory can be references using: dw_tap_data.path

sys.path.append("../notebooks/validation")
import perera
import common

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
from dw_tap.obstacles import AllObstacles
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, MultiPoint, shape
from shapely.ops import split, nearest_points
import fiona
import pickle

import site_index_oe

import warnings
# Hide all warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logger = logging.getLogger()

obstacle_data_dir = os.path.join(dw_tap_data.path, "01 One Energy Turbine Data/3dbuildings_geojson")

index = site_index_oe.SiteIndexOE()

def test_perera_p3wtg1():
    tid = 'p3wtg1'
    wind_source = "wtk"
    obstacle_mode = "bldgsonly"
    
    wtk_df = pd.read_csv(os.path.join(dw_tap_data.path, "01 One Energy Turbine Data/wtk_tp.csv.bz2"))
    
    # Create dict with dataframes that correspond to selected tid's
    dfs_by_tid = {}
    dfs_by_tid[tid] = wtk_df[wtk_df["tid"] == tid].reset_index(drop=True)
    
    atmospheric_input = dfs_by_tid
    
    obs = AllObstacles(data_dir=dw_tap_data.path, types=["oneenergy"], debug=False)
    obs_subset = obs.get("oneenergy", tid, obstacle_mode)
    
    pid = tid[:2]
    
    dest_file = "%s/%s_epsg3740.json" % (obstacle_data_dir, tid)
    obs_subset.to_crs(3740).to_file(dest_file, driver="GeoJSON", index=False)
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:3740")
    buildings = fiona.open("%s/%s_epsg3740.json" % (obstacle_data_dir, tid))
    
    features = {"treesasbldgs": {}}
    
    row = index.lookup_by_tid(tid)
    lat = row["Latitude"]
    lon = row["Longitude"]
    lat,lon = transformer.transform(lat,lon)
    point = Point(lat,lon)
    features["treesasbldgs"][tid] = perera.calculate_perera_features(point,buildings)
    
    #pickle.dump( features["treesasbldgs"], open( "%s/%s_perera_treesasbldgs.p" % (obstacle_data_dir, tid), "wb" ) )
   
    df = atmospheric_input[tid]
 
    df["sector"] = common.sectorize(df["ws"])
    
    df["ws-adjusted"] = \
        df[["tid","sector","ws"]].\
        apply(perera.perera,axis=1,args=(features["treesasbldgs"],))
    
    df["ws-adjusted-2"] = \
        df[["tid","sector","ws"]].\
        apply(perera.perera2,axis=1,args=(features["treesasbldgs"],))
    
    df["ws-adjusted-3"] = \
        df[["tid","sector","ws"]].\
        apply(perera.perera3,axis=1,args=(features["treesasbldgs"],))
 
    p1min = df["ws-adjusted"].min()
    p1max = df["ws-adjusted"].max()
    assert p1min >= 0, "Testing Perera1: min ws-adjusted should be non-negative (observed: %f)." % p1min
    assert p1max <= 150, "Testing Perera1: realistic max ws-adjusted should be <= 150 m/s (observed: %f)." % p1max
    
    p2min = df["ws-adjusted-2"].min()
    p2max = df["ws-adjusted-2"].max()
    assert p2min >= 0, "Testing Perera2: min ws-adjusted should be non-negative (observed: %f)." % p2min
    assert p2max <= 150, "Testing Perera2: realistic max ws-adjusted should be <= 150 m/s (observed: %f)." % p2max
    
    p3min = df["ws-adjusted-3"].min()
    p3max = df["ws-adjusted-3"].max()
    assert p3min >= 0, "Testing Perera3: min ws-adjusted should be non-negative (observed: %f)." % p3min
    assert p3max <= 150, "Testing Perera3: realistic max ws-adjusted should be <= 150 m/s (observed: %f)." % p3max
 
 
