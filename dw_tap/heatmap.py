import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import geometry

def gen_heatmap_points(obstacles, 
                      box_width_m = 300,
                      box_height_m = 300,
                      n_points = 20,
                      file_dest="",
                      visualize=False):

    obstacles_centroid = obstacles.dissolve().to_crs('+proj=cea').centroid.to_crs(obstacles.crs)
    
    # Use meters-per-degree approximation for a perfect sphere to go from meters to degress
    m_per_deg = 111000

    # halfs of box / dimensions
    lon_delta = (box_width_m / m_per_deg ) / 2.0
    lat_delta = (box_height_m / m_per_deg ) / 2.0

    lon_n, lat_n = (n_points, n_points)
    x = np.linspace(obstacles_centroid.geometry.x-lon_delta, 
                    obstacles_centroid.geometry.x+lon_delta, 
                    lon_n)
    y = np.linspace(obstacles_centroid.geometry.y-lat_delta, 
                    obstacles_centroid.geometry.y+lat_delta, 
                    lat_n)
    xv, yv = np.meshgrid(x, y)

    gridpoints = gpd.GeoDataFrame(
        {"geometry": [geometry.Point(el) for el in zip(xv.flatten(), yv.flatten())]})
    
    gridpoints["obstacleRel"] = "outside"
    inside_idx = gpd.sjoin(gridpoints, obstacles, op='within').index
    for idx in inside_idx:
        gridpoints.at[idx, "obstacleRel"] = "inside"

    if file_dest:
        gridpoints.to_file(file_dest, driver="GeoJSON")  
        
    if visualize:
        ax=obstacles.geometry.plot()
        gridpoints[gridpoints["obstacleRel"] == "outside"].plot(color="green", ax=ax)
        plt.show()

    return gridpoints