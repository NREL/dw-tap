# -*- coding: utf-8 -*-
import numpy as np
import scipy
import geopandas as gpd
import matplotlib
import csv
import math
import pyproj
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from scipy import interpolate
import warnings
from pyproj import Proj, transform
from shapely.geometry import Polygon, Point

# work with Geojson files:
def geojson_toCoordinate(df_places, minx, maxx, miny, maxy, trees, porosity):
    H = []
    XY = []
    eps = []
    i = 0
    for place in df_places.geometry: 
        if (( trees == True)):
            #if ((place.bounds[0]>minx) and (place.bounds[0]<maxx)): #100
                #if ((place.bounds[1]>miny) and (place.bounds[1]<maxy)): #100
            if np.isfinite(df_places.height[i]):
                    if (df_places.height[i]!='null'):
                        XY.append(np.array([_LatLon_To_XY(lat, lon) for lon, lat in zip(*place.exterior.coords.xy)]))
                        H.append(df_places.height[i])
                        if (( trees == True)):
                            # dmdu fix: No Type, Buildings only
                            eps.append(0.0)
        if (( trees == False)):
            
            #if df_places['type'][i] == 'building':
            # dmdu fix:
            if True:
                #if ((place.bounds[0]>minx) and (place.bounds[0]<maxx)): #100
                    #if ((place.bounds[1]>miny) and (place.bounds[1]<maxy)): #100
                if np.isfinite(df_places.height[i]):
                    if (df_places.height[i]!='null'):
                        XY.append(np.array([_LatLon_To_XY(lat, lon) for lon, lat in zip(*place.exterior.coords.xy)]))
                        H.append(df_places.height[i])
                        eps.append(0.0)
        i+=1
    return XY, H, eps

def prepare_data(XY,XYt,theta):
    XYp=[]
    XYr=[]
    xyc=[]
    xci=[]
    yci=[]
    Li=[]
    Wi=[]
    XYti=[]
    for i in range(len(XY)):
        xc, yc = _find_centroid(XY[i])
        #print('size place=', len(XY))
        xyc=_transform_tocent(XY[i], xc, yc)
        xyr=_rotate_coord(xyc, theta)
        L,W =_find_LW(xyr, np.shape(theta)[0])
        #print(L,W)
        xtr1=[]
        for j in range (len(XYt)):
            #print(turbine)
            xyti = _transform_tocentT(XYt[j], xc, yc)
            #print(xyti)
            #print(len(xyti))
            xtr=_rotate_coordT(xyti, theta)
            xtr1.append(xtr)
        XYti.append(xtr1)
        xci.append(xc)
        yci.append(yc)
        XYp.append(xyc)
        XYr.append(xyr)
        Li.append(L)
        Wi.append(W)
    return xci, yci, XYp, XYr, Li, Wi, XYti
#centroid x, centroid y, transformed XYp[x,y], rotated XYr[theta][x,y], L[theta], W[theta],XYti

def _find_centroid(XY):
    xc=(XY[:,0].max()+XY[:,0].min())/2.
    yc=(XY[:,1].max()+XY[:,1].min())/2.
    #print('xc: ', xc)
    return xc, yc

def _transform_tocent(XY, xc, yc):
    XYp=np.empty_like(XY)
    XYp[:,0]=XY[:,0]-xc
    XYp[:,1]=XY[:,1]-yc
    #print('XYp output: \n', XYp)
    return XYp

def _rotate_coord(XY, theta):
    XYt=np.empty([np.shape(theta)[0],np.shape(XY)[0], np.shape(XY)[1]])
    for i in range (np.size(theta)):
        XYt[i,:,0]=XY[:,0]*np.cos(np.deg2rad(theta[i]))-XY[:,1]*np.sin(np.deg2rad(theta[i]))
        XYt[i,:,1]=XY[:,0]*np.sin(np.deg2rad(theta[i]))+XY[:,1]*np.cos(np.deg2rad(theta[i]))
    return XYt.T

def _find_LW(XY, n):
    L = np.empty(n)
    W = np.empty(n)
    for i in range (n):
        W[i]=XY[0,:,i].max()-XY[0,:,i].min()
        L[i]=XY[1,:,i].max()-XY[1,:,i].min()
    return L, W

def _transform_tocentT(XY, xc, yc):
    XYp=np.empty_like(XY)
    if XY.ndim>1:
        XYp[:,0]=XY[:,0]-xc
        XYp[:,1]=XY[:,1]-yc
    else:
        XYp[0]=XY[0]-xc
        XYp[1]=XY[1]-yc
    return XYp

def _rotate_coordT(XY, theta):
    XYt=np.empty([np.shape(theta)[0],2])
    for i in range (np.size(theta)):
           XYt[i,0]=XY[0]*np.cos(np.deg2rad(theta[i]))-XY[1]*np.sin(np.deg2rad(theta[i]))
           XYt[i,1]=XY[0]*np.sin(np.deg2rad(theta[i]))+XY[1]*np.cos(np.deg2rad(theta[i]))
    return XYt


def _removeNegative(u):
    c= abs(u)+u
    return c/2


def _removeNans(u,s, b, x):
    v=[]
    w=[]
    p=[]
    d=[]

    i=0
    for up in u:
        if ((not math.isnan(u[i])) & (not math.isnan(b[i]))) :
            v.append(u[i])
            w.append(s[i])
            p.append(b[i])
            d.append(x[i])

        i+=1
    return v, w, p, d

def _removeNans1(u,s, b, x,f,g):
    v=[]
    w=[]
    p=[]
    d=[]
    k=[]
    l=[]
    i=0
    for up in u:
        if ((not math.isnan(u[i])) & (not math.isnan(b[i])) & (not math.isnan(f[i]))& (g[i]>0.)):
            v.append(u[i])
            w.append(s[i])
            p.append(b[i])
            d.append(x[i])
            k.append(f[i])
            l.append(g[i])
        i+=1
    return v, w, p, d, k,l

def _removeNans2(u,s, b, x,f,g):
    v=[]
    w=[]
    p=[]
    d=[]
    k=[]
    l=[]
    i=0
    for up in u:
        if ( (u[i]>2.0) & (u[i]<12.0)):
            v.append(u[i])
            w.append(s[i])
            p.append(b[i])
            d.append(x[i])
            k.append(f[i])
            l.append(g[i])
        i+=1
    return v, w, p, d, k,l

def _maskeddata(u, v, x):
    um = []
    vm = []
    i=0
    for xp in x:

        if xp < 0.:
            um.append(u[i])
            vm.append(v[i])
        i+=1
    return um, vm

def _maskeddata1(u, v, x):
    um = []
    vm = []
    i=0
    for xp in x:
        if xp > 0.:
            #if (((u[i] -u[i]) == 0.0) &&  ((v[i] -v[i]) == 0.0)):
             if ((not math.isnan(u[i]) ) &  (not math.isnan(v[i]))):
                um.append(u[i])
                vm.append(v[i])
        i+=1
    return um, vm

def _LatLon_To_XY(Lat,Lon):
    """
    Input: Lat, Lon coordinates in degrees. 
    _LatLon_To_XY uses the albers projection to transform the lat lon coordinates in degrees to meters. 
    This is an internal function called by get_candidate.
    Output: Meter representation of coordinates. 
    """
    P = pyproj.Proj("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
    
    return P(Lon, Lat) #returned x, y note: lon has to come first here when calling

def wd_to_bin_avg(v, wd_bins):
    """ Maps wind direction to a bin average given a set of wind direction bin limits """
    for b in wd_bins:
        lb, hb = b[0], b[1]
        if v >= lb and v < hb:
            return (lb + hb) / 2.0
    if v == wd_bins[-1][1]:
        lb, hb = wd_bins[-1][0], wd_bins[-1][1]
        return (lb + hb) / 2.0
    raise ValueError("Given wind direction value (%.3f) doesn't fit any of the created bins" % v)
    return np.nan

def ws_to_bin_avg(v, ws_bins):
    """ Maps wind speed to a bin average given a set of wind speed bin limits """
    for b in ws_bins:
        lb, hb = b[0], b[1]
        if v >= lb and v < hb:
            return (lb + hb) / 2.0
    if v == ws_bins[-1][1]:
        lb, hb = ws_bins[-1][0], ws_bins[-1][1]
        return (lb + hb) / 2.0
    raise ValueError("Given wind speed value (%.3f) doesn't fit any of the created bins" % v)
    return np.nan

def wind_binning(atmospheric_df, wd_bin_width=10.0, ws_bin_width=1.0):
    """ Maps a dataframe with wind data to bins; outputs two dataframes: timeseries (with row-wise references to bins) and bins dataframe"""

    wd_bin_stops = np.linspace(0, 360, int(360 / wd_bin_width) + 1, endpoint=True)
    wd_bins = [(wd_bin_stops[i], wd_bin_stops[i+1]) for i in range(len(wd_bin_stops)-1)]

    ws_max = math.ceil(atmospheric_df["ws"].max())
    ws_bin_stops = np.linspace(0, ws_max, int(ws_max / ws_bin_width) + 1, endpoint=True)
    ws_bins = [(ws_bin_stops[i], ws_bin_stops[i+1]) for i in range(len(ws_bin_stops)-1)]
    
    atmospheric_df_binned = atmospheric_df.copy()
    atmospheric_df_binned["wd"] = atmospheric_df_binned["wd"].apply(lambda x: ws_to_bin_avg(x, wd_bins))
    atmospheric_df_binned["ws"] = atmospheric_df_binned["ws"].apply(lambda x: ws_to_bin_avg(x, ws_bins))
    #return atmospheric_df_binned
    
    atmospheric_df_binned["bin_idx"] = -1
    grouped = atmospheric_df_binned.groupby(["ws", "wd"])

    # Create dataframe with bin averages for ws and wd and 
    # make the rows in the original dataframe with timeseries refer to the corresponding bin rows

    atmospheric_bins = pd.DataFrame(columns=["ws", "wd"], index=range(len(grouped)))
    for bin_idx, (idx, grp) in enumerate(grouped):
        atmospheric_bins.iloc[bin_idx] = list(idx)
        for grp_idx, row in grp.iterrows():
            atmospheric_df_binned.at[grp_idx, "bin_idx"] = bin_idx
    
    # For bins df, use first datetime
    atmospheric_bins["datetime"] = atmospheric_df_binned["datetime"].tolist()[0]
    # For bins df, use average inversemoninobukhovlength_2m
    atmospheric_bins["inversemoninobukhovlength_2m"] = atmospheric_df_binned["inversemoninobukhovlength_2m"].mean()
        
    if atmospheric_df_binned["bin_idx"].min() < 0:
        # i.e., if there are -1 present (unmapped rows):
        raise ValueError("Possible issue with data binning: at least one row has not been mapped to bin averages")
            
    return atmospheric_df_binned, atmospheric_bins

def filter_obstacles(tid, 
                     raw_obstacle_df, 
                     include_trees=True,
                     min_height_thresh=1.0,
                     turbine_height_for_checking=None,
                     limit_to_radius_in_m=False,
                     turbine_lat_lon=None,
                     version=2):
    """ Process obstacle data in preparation for running LOMs. """
    
#     if version == 2:
#         # Older version, the one with medians and maximums
        
#         # Start with a copy of the given df
#         df = raw_obstacle_df.copy()

#         # 3DBuildings data have only buildings records and None is listed under feature_type column
#         df["feature_type"] = df["feature_type"].map({None: "building", 
#                                                      np.NaN: "building",
#                                                      "building": "building", # leave unchanged
#                                                      "tree": "tree"}) # leave unchanged

#         # Iterate over rows and decide on the height column data for different cases
#         if ("height_median" in df.columns) and ("height_max" in df.columns):
#         # if height_median and height_max aren't there, this filter_obstacles() was already applied, at least once, which is acceptable 
#         # for filtering in stages

#             for idx, row in df.iterrows():
#                 if row["feature_type"] == "building":
#                     df.at[idx, "height"] = row["height_median"]
#                 elif row["feature_type"] == "tree":
#                     df.at[idx, "height"] = row["height_max"]
#                 else:
#                     raise ValueError('Unsupported value under feature_type. Row:\n%s' % str(row))

#          # Exclude obstacles with height < min_height_thresh
#         df = df[df["height"] >= min_height_thresh].reset_index(drop=True)

#         # Exclude trees if the argument calls for it
#         if not include_trees:
#             df = df[df["feature_type"] != "tree"].reset_index(drop=True)

#         if turbine_height_for_checking:
#             if len(df[df["height"] >= turbine_height_for_checking]) > 0:
#                 warnings.warn("(tid: %s) Detected at least 1 obstacle that is as tall as the studied turbine:\n%s" % \
#                               (tid, str(df[df["height"] >= turbine_height_for_checking][["height", "feature_type", "geometry"]])))

#         if limit_to_radius_in_m and (limit_to_radius_in_m > 0) and (type(turbine_lat_lon) is tuple):
#             lat, lon = turbine_lat_lon
#             inProj = Proj(init='epsg:4326')
#             outProj = Proj(init='epsg:3857') # Projection with coordinates in meters
#             x,y = transform(inProj, outProj, lon, lat)
#             turbine_point = Point(x,y)
#             turbine_point_buffer = turbine_point.buffer(limit_to_radius_in_m)

#             # Exclude obstacles that don't overlap with the buffer zone at all
#             df = df[~df.to_crs('epsg:3857').intersection(turbine_point_buffer).is_empty].reset_index(drop=True)

#         # Return obstacle dataframe with a subset of columns rather than all
#         return df[["height", "geometry", "feature_type"]]
    
#    elif version == 3:

    if version == 3:
        
        # Newer version, the one with gt2 (>=2m) statistics, and handing building, tree, canopy, and hedgerow feature_types
        
        # This version will use height_median_gt2 and height_percentile_95_gt2 (instead of simple median and max)
        
        # Start with a copy of the given df
        df = raw_obstacle_df.copy()
        
        # Make a copy of a column to preserve the original feature_type
        df["feature_type_raw"] = df["feature_type"]
        
        # Initial feature_type mapping; 3DBuildings data have only buildings records and None is listed under feature_type column;
        df["feature_type"] = df["feature_type"].apply(lambda x: "building" if not x else x) # replace x if x is None or np.NaN

        # Iterate over rows and decide on the height column data for different cases
        if ("height_median_gt2" in df.columns) and ("height_percentile_95_gt2" in df.columns):
        # if height_median_gt2 and height_percentile_95_gt2 aren't there, this filter_obstacles() was already applied, at least once, which is acceptable 
        # for filtering in stages

            for idx, row in df.iterrows():
                if row["feature_type"] == "building":
                    df.at[idx, "height"] = row["height_median_gt2"]
                elif row["feature_type"] == "canopy":
                    df.at[idx, "height"] = row["height_median_gt2"]
                elif row["feature_type"] == "hedgerow":
                    df.at[idx, "height"] = row["height_median_gt2"]
                elif row["feature_type"] == "tree":
                    df.at[idx, "height"] = row["height_percentile_95_gt2"]
                else:
                    raise ValueError('Unsupported value under feature_type. Row:\n%s' % str(row))
                    
        # Additional feature_type mapping
        df["feature_type"] = df["feature_type"].map({"building": "building", 
                                                     "tree": "tree",
                                                     "canopy": "tree",
                                                     "hedgerow": "tree"}) 

         # Exclude obstacles with height < min_height_thresh
        df = df[df["height"] >= min_height_thresh].reset_index(drop=True)

        # Exclude trees if the argument calls for it
        if not include_trees:
            df = df[df["feature_type"] != "tree"].reset_index(drop=True)

        if turbine_height_for_checking:
            if len(df[df["height"] >= turbine_height_for_checking]) > 0:
                warnings.warn("(tid: %s) Detected at least 1 obstacle that is as tall as the studied turbine:\n%s" % \
                              (tid, str(df[df["height"] >= turbine_height_for_checking][["height", "feature_type", "geometry"]])))

        if limit_to_radius_in_m and (limit_to_radius_in_m > 0) and (type(turbine_lat_lon) is tuple):
            lat, lon = turbine_lat_lon
            inProj = Proj(init='epsg:4326')
            outProj = Proj(init='epsg:3857') # Projection with coordinates in meters
            x,y = transform(inProj, outProj, lon, lat)
            turbine_point = Point(x,y)
            turbine_point_buffer = turbine_point.buffer(limit_to_radius_in_m)

            # Exclude obstacles that don't overlap with the buffer zone at all
            df = df[~df.to_crs('epsg:3857').intersection(turbine_point_buffer).is_empty].reset_index(drop=True)

        # Return obstacle dataframe with a subset of columns rather than all
        return df[["height", "geometry", "feature_type", "feature_type_raw"]]

    else:
        raise ValueError("Specified version is unsupported.")
    