import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import Point

def locate_nearest_obs_sites(obs_sites_src, lat, lon, height, km_thresh=100, row_lim=3):
    """ Find nearest observational sites and distplay minimal and clear dataframe sorted by dist and height diff.
    
    Usage example: 
        locate_nearest_obs_sites("./met_tower_obs_summary.geojson", 42.0, -84.0, 50)
    """
    
    obs = gpd.read_file(obs_sites_src)
    obs_in_meters = obs.to_crs('epsg:3740')

    p = Point(lat, lon) # (lat, lon is the way to go!)

    # Define the source and target CRS
    source_crs = pyproj.CRS("EPSG:4326")  
    # WGS 84
    target_crs = pyproj.CRS("EPSG:3740")  #3740
    # Web Mercator
    # Create a transformer object
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs)
    # Transform the point
    p_in_meters = Point(transformer.transform(p.x, p.y)) # pyproj needs lat, lon!

    obs_in_meters["Distance from selected site, km"] = obs_in_meters.distance(p_in_meters) / 1000.0

    nearby_obs = obs_in_meters[obs_in_meters["Distance from selected site, km"] <= km_thresh].copy()
    nearby_obs["heigh diff, m"] = np.abs(nearby_obs["height"] - height)
    res = nearby_obs.sort_values(["Distance from selected site, km", "heigh diff, m"], ascending=True)

    # Subsetting rows to the specified limit
    if row_lim:
        final_res = res[:row_lim].copy()
    else:
        final_res = res.copy()

    final_res["Coverage"] = (final_res["n_samples"]/1000.0).round(1).astype(str) + "K datapoints (" + \
        final_res["time_start"].dt.strftime('%b-%Y').astype(str) + "--" + final_res["time_end"].dt.strftime('%b-%Y').astype(str) + ")"

    final_res["height"] = final_res["height"].astype(str) + " (different by: " + final_res["heigh diff, m"].astype(str) + ")"
    
    final_res = final_res.drop(columns=["site_id", "geometry", "time_start", "time_end", "n_samples", "heigh diff, m"])
    final_res["wind_speed_mean"] = final_res["wind_speed_mean"].round(2)
    final_res["Distance from selected site, km"] = final_res["Distance from selected site, km"].round(1)

    final_res.rename(columns={"wind_speed_mean": "Avg. observed windspeed, m/s",\
                              "height": "Observations height, m (difference from selected height, m)", \
                              "type": "Type of site with observations"}, \
                     inplace=True)
    
    return final_res
