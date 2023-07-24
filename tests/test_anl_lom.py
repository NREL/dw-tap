import numpy as np
import pandas as pd
import h5pyd
import geopandas as gpd
from dw_tap.lom import run_lom

# Note: paths below are changed from the notebook code to start from the project's root directory

def test_site_180_close():
    """ This test checks a realistic site, with the turbine placed near the buildings"""
    
    z_turbine = 30 # turbine height in [m]
    lat, lon = 39.32856, -89.40238
    obstacle_file = "sites/180-5BuildingsManual.geojson"

    #Read in data from CSV file instead of hsds + getData
    atmospheric_df = pd.read_csv("data/180_1year_12hourgranularity.csv")

    # Temporary (later need to resave csv without index column) 
    if "Unnamed: 0" in atmospheric_df.columns:
        atmospheric_df.drop(columns=["Unnamed: 0"], inplace=True)

    obstacles_df = gpd.read_file(obstacle_file)
    # Leave in only relevant columns
    obstacles_df = obstacles_df[["height", "geometry"]]

    xy_turbine = [np.array([lon, lat])]

    predictions_df = \
        run_lom(atmospheric_df, obstacles_df, xy_turbine, z_turbine,
                check_distance=True)
    
    assert len(predictions_df) == len(atmospheric_df), \
        "len(predictions_df) should match len(atmospheric_df)"
    
    assert "ws-adjusted" in predictions_df.columns, \
        "Output dataframe should include ws-adjusted column"
    
    assert predictions_df["ws-adjusted"].isna().sum() == 0, \
        "Output should not include NAN values"
    
    assert predictions_df["ws-adjusted"].min() >= 0, \
        "Output should not include negative values"

def test_site_180_far():
    """ This test checks the case with the turbine that is *too* far from the buildings"""
    
    z_turbine = 30 # turbine height in [m]
    lat, lon = 39.32856 + 1.0, -89.40238 # 1 degree difference moves the turbine far enough from the buildings 
    obstacle_file = "sites/180-5BuildingsManual.geojson"

    #Read in data from CSV file instead of hsds + getData
    atmospheric_df = pd.read_csv("data/180_1year_12hourgranularity.csv")

    # Temporary (later need to resave csv without index column) 
    if "Unnamed: 0" in atmospheric_df.columns:
        atmospheric_df.drop(columns=["Unnamed: 0"], inplace=True)

    obstacles_df = gpd.read_file(obstacle_file)
    # Leave in only relevant columns
    obstacles_df = obstacles_df[["height", "geometry"]]

    xy_turbine = [np.array([lon, lat])]

    predictions_df = \
        run_lom(atmospheric_df, obstacles_df, xy_turbine, z_turbine,
                check_distance=True)
    
    # base checks are the same as test_site_180_close 
    
    assert len(predictions_df) == len(atmospheric_df), \
        "len(predictions_df) should match len(atmospheric_df)"
    
    assert "ws-adjusted" in predictions_df.columns, \
        "Output dataframe should include ws-adjusted column"
    
    assert predictions_df["ws-adjusted"].isna().sum() == 0, \
        "Output should not include NAN values"
    
    assert predictions_df["ws-adjusted"].min() >= 0, \
        "Output should not include negative values"
    
    # Additional, specific test in this case (tubine too far --> no impact):
   
    assert (predictions_df["ws-adjusted"] - predictions_df["ws"]).max() == 0, \
        "Output should match input windspeed for turbines that are too far"

def test_site_180_high():
    """ This test checks a realistic site, with the turbine placed near the buildings but very high"""
    
    z_turbine = 100 # turbine height in [m]
    lat, lon = 39.32856, -89.40238
    obstacle_file = "sites/180-5BuildingsManual.geojson"

    #Read in data from CSV file instead of hsds + getData
    atmospheric_df = pd.read_csv("data/180_1year_12hourgranularity.csv")

    # Temporary (later need to resave csv without index column) 
    if "Unnamed: 0" in atmospheric_df.columns:
        atmospheric_df.drop(columns=["Unnamed: 0"], inplace=True)

    obstacles_df = gpd.read_file(obstacle_file)
    # Leave in only relevant columns
    obstacles_df = obstacles_df[["height", "geometry"]]

    xy_turbine = [np.array([lon, lat])]

    predictions_df = \
        run_lom(atmospheric_df, obstacles_df, xy_turbine, z_turbine,
                check_distance=True)
    
    # base checks are the same as test_site_180_close 
    
    assert len(predictions_df) == len(atmospheric_df), \
        "len(predictions_df) should match len(atmospheric_df)"
    
    assert "ws-adjusted" in predictions_df.columns, \
        "Output dataframe should include ws-adjusted column"
    
    assert predictions_df["ws-adjusted"].isna().sum() == 0, \
        "Output should not include NAN values"
    
    assert predictions_df["ws-adjusted"].min() >= 0, \
        "Output should not include negative values"
    
    assert (predictions_df["ws"] - predictions_df["ws-adjusted"]).min() >= 0, \
        "Output should not exceed input wind speed values"
    
    # Additional, specific test in this case (tubine too high --> almost no impact):

    # 0.1 limit will allow some numerical differences at the specified height but the differences should be small
    assert (predictions_df["ws-adjusted"] - predictions_df["ws"]).max() < 0.1, \
        "Output should be very close to the input"


