# You might need to run:
# pip install h5pyd
# before running this notebook
# Another requirement:
# You have to have a ~/.hscfg file (hidden file in your home dir) with the following contents (exclude """):
"""
hs_endpoint = https://tap-hsds.ace.nrel.gov
hs_username = None
hs_password = None
hs_api_key = None
hs_bucket = nrel-pds-hsds
""";

# (this is an HSDS instnace that is dedicated to TAP)

# For additional info refer to this page (example notebook for accessing HSDS Wind data without Rex package):
# https://github.com/NREL/hsds-examples/blob/master/notebooks/01_WTK_introduction.ipynb

import h5pyd
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyproj import Proj
import dateutil
import time
from timeit import default_timer as timer
import math
from operator import itemgetter
from scipy.interpolate import griddata
import dw_tap.interpolation.points as points
import dw_tap.interpolation.timeseries as timeseries
import dw_tap.interpolation.interpolation as interpolation
import dw_tap.vector.vector_transformation as transformation

import pickle as pkl
import h5pyd
from tqdm.auto import tqdm

import sys
sys.path.append("../scripts")

# Can't import it here (won't work in AWS env)
# import dw_tap_data

# Can't import it here (won't work in AWS env)
# import dw_tap_data
#from rex.resource_extraction import MultiYearWindX

def getData(f, lat, lon, height,
            method='IDW',
            power_estimate=False,
            inverse_monin_obukhov_length = False,
            start_time_idx=None, end_time_idx=None, time_stride=None,
            srw=False,
            saved_dt=None
           ):
    """
    Horizontally and vertically interpolates wind speed, wind direction, and
    potentially temperature and pressure. Horizontal interpolation for wind
    speed and direction uses inverse distance weight by default. To remove
    interpolation, set method='nearest'. This is computationally wasteful at
    the moment, because we are still getting data for all four grid points, rather
    than getting only the data for the nearest grid point.

    Horizontal interpolation for temperature and pressure uses nearest neighbor.
    Vertical interpolation uses linear method for all variables.

    Input: read in file, latitude, longitude,
           height of potential turbine/candidate turbine,
           horizontal interpolation method,
           boolean indicating whether temperature and
           pressure data will be needed for power estimate.

    Output: (1) If power_estimate is true, returns are wind speed,
            wind direction, datetime, temperature and pressure.
            (2) If power_estimate is false, returns are wind speed,
            wind direction, and datetime. All values horizontally
            and vertically interpolated.

    """
    # Get the datettimes available in the file
    if type(saved_dt)==type(pd.DataFrame()) and len(saved_dt) > 0:
        dt = saved_dt
    else:
        dt = _getDateTime(f)


    # Determine the model data to retrieve based on input parameters
    if (start_time_idx is not None) and (end_time_idx is not None) and (time_stride is not None):
        # All three are specified
        dt=dt.iloc[start_time_idx:end_time_idx+1:time_stride].reset_index(drop=True)
    else:
        start_time_idx = 0
        end_time_idx = len(dt)
        time_stride=1

    point_lat_lon = (lat, lon)

    # grid_points will either be the wtk grid points, or they will be -1
    # point_idx: List of grid index tuples for WTK bounding box starting from bottom left and moving clockwise.
    # dist: List of distances from point to each bounding box corner, ordered from bottom left corner moving clockwise.
    # grid_points: List of meters coordinates of the box starting from bottom left and moving clockwise.
    # x, y: Meter projection of longitude. Meter projection of latitude.
    point_idx, dist, grid_points, x, y = _indicesForCoord(f, point_lat_lon[0], point_lat_lon[1])

    # Get lower and upper heighs available in WTK data for vertical interpolation
    wtk_heights = np.array([10, 40, 60, 80, 100, 120, 140, 160, 200])
    if height in wtk_heights:
        lower_height = height
        upper_height = height
    elif height < 40:
        # I suppose we neglect height of 10 because it is not reliable? -KM
        lower_height = 40
        upper_height = 60
    elif height > 200:
        lower_height = 160
        upper_height = 200
    else:
        lower_height = wtk_heights[wtk_heights < height].max()
        upper_height = wtk_heights[wtk_heights > height].min()

    #if they are -1, it means the lat, lon request was not valid (out of bounds)
    if grid_points == [-1, -1, -1, -1]:
        return pd.DataFrame()

    desired_point = points.XYZPoint(lat, lon, height, 'desired')
    # Won't these always be the same, since we are getting point_idx
    # from bottom-left and going around clockwise? The bottom left should
    # always be minx and miny and the top right should always be max x and max y
    # unless maybe the rectangle is rotated? -KM
    minx = min(point_idx,key=itemgetter(0))[0]
    maxx = max(point_idx,key=itemgetter(0))[0]
    miny = min(point_idx,key=itemgetter(1))[1]
    maxy = max(point_idx,key=itemgetter(1))[1]

    # print(minx, maxx, miny, maxy)
    # print(start_time_idx, end_time_idx)
    # print(lower_height, upper_height)
    # Wind Speed Fetching
    ws_lower = f['windspeed_%sm' % lower_height]
    ws_lower = ws_lower[start_time_idx:end_time_idx+1:time_stride, minx:maxx+1, miny:maxy+1]
    ws_lower_df = pd.DataFrame({'1': ws_lower[:, 0, 0],
                                '2': ws_lower[:, 1, 0],
                                '3': ws_lower[:, 1, 1],
                                '4': ws_lower[:, 0, 1]})
    ws_upper = f['windspeed_%sm' % upper_height]
    ws_upper = ws_upper[start_time_idx:end_time_idx+1:time_stride, minx:maxx+1, miny:maxy+1]
    ws_upper_df = pd.DataFrame({'1': ws_upper[:, 0, 0],
                                '2': ws_upper[:, 1, 0],
                                '3': ws_upper[:, 1, 1],
                                '4': ws_upper[:, 0, 1]})

    #Wind Direction Fetching
    wd_lower = f['winddirection_%sm' % lower_height]
    wd_lower = wd_lower[start_time_idx:end_time_idx+1:time_stride, minx:maxx+1, miny:maxy+1]
    wd_lower_df = pd.DataFrame({'1': wd_lower[:, 0, 0],
                                '2': wd_lower[:, 1, 0],
                                '3': wd_lower[:, 1, 1],
                                '4': wd_lower[:, 0, 1]})
    wd_upper = f['winddirection_%sm' % upper_height]
    wd_upper = wd_upper[start_time_idx:end_time_idx+1:time_stride, minx:maxx+1, miny:maxy+1]
    wd_upper_df = pd.DataFrame({'1': wd_upper[:, 0, 0],
                                '2': wd_upper[:, 1, 0],
                                '3': wd_upper[:, 1, 1],
                                '4': wd_upper[:, 0, 1]})

    # Convert wind direction mathematical degrees to met degrees
    wd_lower_df = wd_lower_df.apply(transformation._convert_to_met_deg,
                                    args=(), axis=1)
    wd_upper_df = wd_upper_df.apply(transformation._convert_to_met_deg,
                                    args=(), axis=1)

    # Break down windspeed into orthogonal components for interpolation
    # We will apply the interpolation process to the orthogonal components,
    # then combine the results into a final, interpolated vector.
    # Getting the 'u' wind speed vector for lower and upper heights
    u_lower, u_upper = transformation._convert_to_vector_u(wd_lower_df,
                                                           wd_upper_df,
                                                           ws_lower_df,
                                                           ws_upper_df)
    u_lower["spatially_interpolated"] = u_lower.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,method), axis=1)
    u_upper["spatially_interpolated"] = u_upper.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,method), axis=1)
    u_lower = pd.Series(u_lower["spatially_interpolated"], name='wd')
    u_upper = pd.Series(u_upper["spatially_interpolated"], name='wd')
    u_final = _interpolate_vertically(lat, lon,
                                      u_lower, u_upper,
                                      height, desired_point,
                                      "polynomial")

    # Getting the 'v' wind speed vector for lower and upper heights
    v_lower, v_upper = transformation._convert_to_vector_v(wd_lower_df,
                                                           wd_upper_df,
                                                           ws_lower_df,
                                                           ws_upper_df)
    v_lower["spatially_interpolated"] = v_lower.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,method), axis=1)
    v_upper["spatially_interpolated"] = v_upper.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,method), axis=1)
    v_lower = pd.Series(v_lower["spatially_interpolated"], name='wd')
    v_upper = pd.Series(v_upper["spatially_interpolated"], name='wd')
    v_final = _interpolate_vertically(lat, lon,
                                      v_lower, v_upper,
                                      height, desired_point,
                                      "polynomial")

    # Combining the interpolated 'u' and 'v' vectors into an interpolated
    # wind speed and wind direction
    ws_result = transformation._convert_to_ws(u_final, v_final)
    wd_result = transformation._convert_to_degrees(u_final, v_final)
    wd_result = wd_result.apply(transformation._convert_to_math_deg, args=())

    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_result
    df["wd"] = wd_result

    if power_estimate:
        dtemp = f['temperature_%sm' % lower_height]
        dtemp = dtemp[start_time_idx:end_time_idx+1:time_stride,
                      minx:maxx+1, miny:maxy+1]
        dtemp = pd.DataFrame({'1': dtemp[:,0,0], '2': dtemp[:,1,0], '3': dtemp[:,1,1], '4': dtemp[:,0,1]})
        dtemp1 = f['temperature_%sm' % upper_height]
        dtemp1 = dtemp1[start_time_idx:end_time_idx+1:time_stride,
                        minx:maxx+1, miny:maxy+1]
        dtemp1 = pd.DataFrame({'1': dtemp1[:,0,0], '2': dtemp1[:,1,0], '3': dtemp1[:,1,1], '4': dtemp1[:,0,1]})
        dtemp = dtemp.apply(_interpolate_spatially_row, args=(dist, grid_points, x, y, 'nearest'), axis=1)
        dtemp1 = dtemp1.apply(_interpolate_spatially_row, args=(dist, grid_points, x, y, 'nearest'), axis=1)
        dtemp_final = _interpolate_vertically(lat, lon, dtemp, dtemp1, height, desired_point, "polynomial")

        dpres0 = f['pressure_0m']
        dpres0 = dpres0[start_time_idx:end_time_idx+1:time_stride,
                        minx:maxx+1, miny:maxy+1]
        dpres0 = pd.DataFrame({'1': dpres0[:,0,0], '2': dpres0[:,1,0], '3': dpres0[:,1,1], '4': dpres0[:,0,1]})
        dpres100 = f['pressure_100m']
        dpres100 = dpres100[start_time_idx:end_time_idx+1:time_stride,
                            minx:maxx+1, miny:maxy+1]
        dpres100 = pd.DataFrame({'1':dpres100[:,0,0],'2':dpres100[:,1,0],'3':dpres100[:,1,1],'4':dpres100[:,0,1]})
        dpres0 = dpres0.apply(_interpolate_spatially_row, args=(dist, grid_points, x, y, method), axis=1)
        dpres100 = dpres100.apply(_interpolate_spatially_row, args=(dist, grid_points, x, y, method), axis=1)
        dpres_final = _interpolate_vertically(lat, lon, dpres0, dpres100, height, desired_point, "polynomial")

        df["temp"] = dtemp_final
        df["pres"] = dpres_final

    if inverse_monin_obukhov_length:
        dimol = f['inversemoninobukhovlength_2m']
        dimol = dimol[start_time_idx:end_time_idx+1:time_stride,
                      minx:maxx+1, miny:maxy+1]
        dimol = pd.DataFrame({'1': dimol[:,0,0], '2': dimol[:,1,0], '3': dimol[:,1,1], '4': dimol[:,0,1]})
        dimol_final = dimol.apply(_interpolate_spatially_row, args=(dist, grid_points, x, y, 'IDW'), axis=1)
        df["inversemoninobukhovlength_2m"] = dimol_final

    if srw:
        df["temp"] = df["temp"] - 273.15
        df["pres"] = df["pres"]*0.00000986923

    return df


def get_wtk_data_idw(f, lat, lon, height,
               start_time=None, end_time=None, time_stride=None):

    def power_law(ws_lower, ws_upper, height_lower, height_upper, height):
        # Use default alpha if ws_lower and ws_upper are opposite directions, or if either ws == 0
        alpha = np.where(
            (ws_lower * ws_upper <= 0),
            1/7.0,
            np.log(ws_lower/ws_upper) / np.log(height_lower/height_upper)
        )
        ws = ws_upper * ((height / height_upper) ** alpha)
        return ws

    # Get the datettimes available in the file
    dt = _getDateTime(f)


    # Determine the model data to retrieve based on input parameters
    if (start_time is not None) and (end_time is not None) and (time_stride is not None):
        # All three are specified
        dt=dt.loc[(dt >= start_time) & (dt <= end_time)].iloc[::time_stride]
        if len(dt) == 0: # No overlap
            return None

    # grid_points will either be the wtk grid points, or they will be -1
    # point_idx: List of grid index tuples for WTK bounding box starting from bottom left and moving clockwise.
    # dist: List of distances from point to each bounding box corner, ordered from bottom left corner moving clockwise.
    # grid_points: List of meters coordinates of the box starting from bottom left and moving clockwise.
    # x, y: Meter projection of longitude. Meter projection of latitude.
    point_idx, dist, grid_points, x, y = _indicesForCoord(f, lat, lon)

    # Get lower and upper heighs available in WTK data for vertical interpolation
    wtk_heights = np.array([10, 40, 60, 80, 100, 120, 140, 160, 200])
    if height in wtk_heights:
        lower_height = height
        upper_height = height
    elif height < 10:
        # I suppose we neglect height of 10 because it is not reliable? -KM
        lower_height = 10
        upper_height = 10
    elif height > 200:
        lower_height = 160
        upper_height = 200
    else:
        lower_height = wtk_heights[wtk_heights < height].max()
        upper_height = wtk_heights[wtk_heights > height].min()

    #if they are -1, it means the lat, lon request was not valid (out of bounds)
    if grid_points == [-1, -1, -1, -1]:
        return pd.DataFrame()

    desired_point = points.XYZPoint(lat, lon, height, 'desired')
    
    minx = min(point_idx,key=itemgetter(0))[0]
    maxx = max(point_idx,key=itemgetter(0))[0]
    miny = min(point_idx,key=itemgetter(1))[1]
    maxy = max(point_idx,key=itemgetter(1))[1]

    # Wind Speed Fetching
    ws_lower = f['windspeed_%sm' % lower_height]
    ws_lower = ws_lower[dt.index[0]:dt.index[-1]:time_stride, minx:maxx+1, miny:maxy+1]
    ws_lower_df = pd.DataFrame({'1': ws_lower[:, 0, 0],
                                '2': ws_lower[:, 1, 0],
                                '3': ws_lower[:, 1, 1],
                                '4': ws_lower[:, 0, 1]})
    ws_upper = f['windspeed_%sm' % upper_height]
    ws_upper = ws_upper[dt.index[0]:dt.index[-1]:time_stride, minx:maxx+1, miny:maxy+1]
    ws_upper_df = pd.DataFrame({'1': ws_upper[:, 0, 0],
                                '2': ws_upper[:, 1, 0],
                                '3': ws_upper[:, 1, 1],
                                '4': ws_upper[:, 0, 1]})

    #Wind Direction Fetching
    wd_lower = f['winddirection_%sm' % lower_height]
    wd_lower = wd_lower[dt.index[0]:dt.index[-1]:time_stride, minx:maxx+1, miny:maxy+1]
    wd_lower_df = pd.DataFrame({'1': wd_lower[:, 0, 0],
                                '2': wd_lower[:, 1, 0],
                                '3': wd_lower[:, 1, 1],
                                '4': wd_lower[:, 0, 1]})
    wd_upper = f['winddirection_%sm' % upper_height]
    wd_upper = wd_upper[dt.index[0]:dt.index[-1]:time_stride, minx:maxx+1, miny:maxy+1]
    wd_upper_df = pd.DataFrame({'1': wd_upper[:, 0, 0],
                                '2': wd_upper[:, 1, 0],
                                '3': wd_upper[:, 1, 1],
                                '4': wd_upper[:, 0, 1]})

    # Convert wind direction mathematical degrees to met degrees
    wd_lower_df = wd_lower_df.apply(transformation._convert_to_met_deg,
                                    args=(), axis=1)
    wd_upper_df = wd_upper_df.apply(transformation._convert_to_met_deg,
                                    args=(), axis=1)


    # Break down windspeed into orthogonal components for interpolation
    # We will apply the interpolation process to the orthogonal components,
    # then combine the results into a final, interpolated vector.
    # Getting the 'u' wind speed vector for lower and upper heights
    u_lower, u_upper = transformation._convert_to_vector_u(wd_lower_df,
                                                           wd_upper_df,
                                                           ws_lower_df,
                                                           ws_upper_df)
    u_lower["spatially_interpolated"] = u_lower.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,"IDW"), axis=1)
    u_upper["spatially_interpolated"] = u_upper.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,"IDW"), axis=1)
    u_lower = pd.Series(u_lower["spatially_interpolated"], name='wd')
    u_upper = pd.Series(u_upper["spatially_interpolated"], name='wd')
    u_final = power_law(u_lower, u_upper, lower_height, upper_height, height) if lower_height != upper_height else u_lower

    # Getting the 'v' wind speed vector for lower and upper heights
    v_lower, v_upper = transformation._convert_to_vector_v(wd_lower_df,
                                                           wd_upper_df,
                                                           ws_lower_df,
                                                           ws_upper_df)
    v_lower["spatially_interpolated"] = v_lower.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,"IDW"), axis=1)
    v_upper["spatially_interpolated"] = v_upper.apply(_interpolate_spatially_row,
                                                      args=(dist,grid_points,x,y,"IDW"), axis=1)
    v_lower = pd.Series(v_lower["spatially_interpolated"], name='wd')
    v_upper = pd.Series(v_upper["spatially_interpolated"], name='wd')
    v_final = power_law(v_lower, v_upper, lower_height, upper_height, height) if lower_height != upper_height else v_lower

    # Combining the interpolated 'u' and 'v' vectors into an interpolated
    # wind speed and wind direction
    ws_result = transformation._convert_to_ws(u_final, v_final)
    wd_result = transformation._convert_to_degrees(u_final, v_final)
    wd_result = wd_result.apply(transformation._convert_to_math_deg, args=())

    dt.name = "datetime"
    df = pd.DataFrame(dt).reset_index(drop=True)
    df["ws"] = ws_result
    df["wd"] = wd_result
    return df



def get_data_wtk_led_idw(myr_pathstr,
                        lat, lon, height,
                        start_time=None, end_time=None, time_stride=None):
    """
    Vertically interpolates wind speed and wind direction.
    Vertical interpolation uses linear method for all variables.
    Input: path string to get read in file, latitude, longitude, height of potential turbine/candidate turbine
    Output: Pandas DataFrame of wind speed, wind direction, and datetime.
    """

    def power_law(ws_lower, ws_upper, height_lower, height_upper, height):
        # Use default alpha if ws_lower and ws_upper are opposite directions, or if either ws == 0
        alpha = np.where(
            np.isnan(ws_lower) or np.isnan(ws_upper) or (ws_lower * ws_upper <= 0),
            1/7.0,
            np.log(ws_lower/ws_upper) / np.log(height_lower/height_upper)
        )
        ws = ws_upper * ((height / height_upper) ** alpha)
        return ws

    desired_point = points.XYZPoint(lat, lon, height, 'desired')

    wtk_heights = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000])
    if height in wtk_heights:
        if 'ANL_4km_north_america' in myr_pathstr:
            myr = MultiYearWindX(myr_pathstr, hsds=False)
        else:
            myr = MultiYearWindX(myr_pathstr + '_%sm.h5' % height, hsds=False)

        dt = _get_dt(myr, start_time, end_time, time_stride)
        dd, ii = myr.tree.query((lat, lon), 4)

        ws_u = list()
        ws_v = list()
        for idx, d in zip(ii,dd):
            ws = myr['windspeed_%sm' % height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]
            wd = myr['winddirection_%sm' % height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]

            wd = (270 - wd) % 360
            u = pd.Series(-ws * np.sin((math.pi/180) * wd))
            v = pd.Series(-ws * np.cos((math.pi/180) * wd))
            ws_u.append(u)
            ws_v.append(v)

        u_final = np.sum([u / d for u, d in zip(ws_u, dd)], axis=0) / np.sum([1 / d for d in dd])
        v_final = np.sum([v / d for v, d in zip(ws_v, dd)], axis=0) / np.sum([1 / d for d in dd])

        ws_final = transformation._convert_to_ws(u_final, v_final)
        wd_final = transformation._convert_to_degrees(u_final, v_final)
        # Convert wd to meteorological degrees
        wd_final = (270 - wd_final) % 360

        dt = dt.reset_index(drop=True)
        dt.name = "datetime"
        df = pd.DataFrame(dt)
        df["ws"] = ws_final
        df["wd"] = wd_final
        return df
    elif height < 10:
        lower_height = 10
        upper_height = 20
    elif height > 1000:
        lower_height = 500
        upper_height = 1000
    else:
        lower_height = wtk_heights[wtk_heights < height].max()
        upper_height = wtk_heights[wtk_heights > height].min()

    if 'ANL_4km_north_america' in myr_pathstr:
        myr_lower = MultiYearWindX(myr_pathstr, hsds=False)
        myr_upper = MultiYearWindX(myr_pathstr, hsds=False)
    else:
        myr_lower = MultiYearWindX(myr_pathstr + '_%sm.h5' % lower_height, hsds=False)
        myr_upper = MultiYearWindX(myr_pathstr + '_%sm.h5' % upper_height, hsds=False)

    dt = _get_dt(myr_lower, start_time, end_time, time_stride)
    dd, ii = myr_lower.tree.query((lat, lon), 4)

    ws_u_lower = list()
    ws_u_upper = list()
    ws_v_lower = list()
    ws_v_upper = list()
    for idx, d in zip(ii,dd):
        ws_lower = myr_lower["windspeed_%sm" % lower_height, dt.index[0]:dt.index[-1]:time_stride, idx]
        wd_lower = myr_lower["winddirection_%sm" % lower_height, dt.index[0]:dt.index[-1]:time_stride, idx]
        ws_upper = myr_upper["windspeed_%sm" % upper_height, dt.index[0]:dt.index[-1]:time_stride, idx]
        wd_upper = myr_upper["winddirection_%sm" % upper_height, dt.index[0]:dt.index[-1]:time_stride, idx]

        # Convert wd to mathematical degrees
        wd_lower = (270 - wd_lower) % 360
        wd_upper = (270 - wd_upper) % 360

        u_lower = pd.Series(-ws_lower * np.sin((math.pi/180) * wd_lower))
        u_upper = pd.Series(-ws_upper * np.sin((math.pi/180) * wd_upper))
        v_lower = pd.Series(-ws_lower * np.cos((math.pi/180) * wd_lower))
        v_upper = pd.Series(-ws_upper * np.cos((math.pi/180) * wd_upper))

        ws_u_lower.append(u_lower)
        ws_u_upper.append(u_upper)
        ws_v_lower.append(v_lower)
        ws_v_upper.append(v_upper)

    u_lower = np.sum([u / d for u, d in zip(ws_u_lower, dd)], axis=0) / np.sum([1 / d for d in dd])
    u_upper = np.sum([u / d for u, d in zip(ws_u_upper, dd)], axis=0) / np.sum([1 / d for d in dd])
    v_lower = np.sum([v / d for v, d in zip(ws_v_lower, dd)], axis=0) / np.sum([1 / d for d in dd])
    v_upper = np.sum([v / d for v, d in zip(ws_v_upper, dd)], axis=0) / np.sum([1 / d for d in dd])

    u_final = power_law(u_lower, u_upper, lower_height, upper_height, height)
    v_final = power_law(v_lower, v_upper, lower_height, upper_height, height)

    ws_final = transformation._convert_to_ws(u_final, v_final)
    wd_final = transformation._convert_to_degrees(u_final, v_final)
    # Convert wd to meteorological degrees
    wd_final = (270 - wd_final) % 360

    dt = dt.reset_index(drop=True)

    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_final
    df["wd"] = wd_final

    return df


def get_data_era5_idw(ds, lat, lon, height):
    def latlon2era5_box(ds, lat, lon):
        # The following relies on u100 being one of the variables in the dataset
        lats = ds.u100.latitude.values
        lons = ds.u100.longitude.values
        lat_box_idx = np.argpartition(np.abs(lats - lat), 2)[:2]
        lon_box_idx = np.argpartition(np.abs(lons - lon), 2)[:2]
        coords = list()
        for i in range(2):
            for j in range(2):
                lat_idx = lat_box_idx[i]
                lon_idx = lon_box_idx[j]
                pseudo_dist = (lat - lats[lat_idx])**2 + (lon - lons[lon_idx])**2
                coords.append((lat_idx, lon_idx, pseudo_dist))
        return coords

    def power_law(ws10, ws100, height):
        # Use default alpha if ws10 and ws100 are opposite directions
        alpha = np.where(
            np.isnan(ws10) | np.isnan(ws100) | (ws10 * ws100 <= 0),
            1/7.0,
            np.log(ws10/ws100) / np.log(10/100)
        )
        ws = ws100 * ((height / 100) ** alpha)
        return ws

    coords = latlon2era5_box(ds, lat, lon)

    lat_idx, lon_idx, dist = map(list, zip(*coords))

    u10 = np.sum([ds.u10.values[:,coord[0],coord[1]] * 1/coord[2] for coord in coords], axis=0) / np.sum([1/coord[2] for coord in coords])
    v10 = np.sum([ds.v10.values[:,coord[0],coord[1]] * 1/coord[2] for coord in coords], axis=0) / np.sum([1/coord[2] for coord in coords])
    u100 = np.sum([ds.u100.values[:,coord[0],coord[1]] * 1/coord[2] for coord in coords], axis=0) / np.sum([1/coord[2] for coord in coords])
    v100 = np.sum([ds.v100.values[:,coord[0],coord[1]] * 1/coord[2] for coord in coords], axis=0) / np.sum([1/coord[2] for coord in coords])

    tt = ds.u100.time.values
    df = pd.DataFrame({"datetime": tt,
                       "u10": u10, "v10": v10,
                       "u100": u100, "v100": v100
                      })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ws10"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    df["ws100"] = np.sqrt(df["u100"]**2 + df["v100"]**2)
    if isinstance(height, list):
        for height_ in height:
            if height_ not in [10, 100]:
                # Power-law vertical interpolation
                u_final = power_law(u10, u100, height_)
                v_final = power_law(v10, v100, height_)
                df[f"ws{height_}"] = transformation._convert_to_ws(u_final, v_final)
    else:
        if height not in [10, 100]:
            # Power-law vertical interpolation
            u_final = power_law(u10, u100, height)
            v_final = power_law(v10, v100, height)
            df[f"ws{height}"] = transformation._convert_to_ws(u_final, v_final)

    return df


def get_data_bchrrr_idw(myr,
                        lat, lon, height,
                        start_time=None, end_time=None, time_stride=None):
    """
    Input: path string to get read in file, latitude, longitude, height of potential turbine/candidate turbine
    Output: Pandas DataFrame of wind speed, wind direction, and datetime.
    """

    def power_law(ws_lower, ws_upper, height_lower, height_upper, height):
        # Use default alpha if ws_lower and ws_upper are opposite directions, or if either ws == 0
        alpha = np.where(
            np.isnan(ws_lower) | np.isnan(ws_upper) | (ws_lower <= 0) | (ws_upper <= 0),
            1/7.0,
            np.log(ws_lower/ws_upper) / np.log(height_lower/height_upper)
        )
        ws = ws_upper * ((height / height_upper) ** alpha)
        return ws

    desired_point = points.XYZPoint(lat, lon, height, 'desired')

    wtk_heights = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    if height in wtk_heights:
        dt = _get_dt(myr, start_time, end_time, time_stride)
        dd, ii = myr.tree.query((lat, lon), 4)

        ws_u = list()
        ws_v = list()
        for idx, d in zip(ii,dd):
            ws = myr['windspeed_%sm' % height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]
            wd = myr['winddirection_%sm' % height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]

            wd = (270 - wd) % 360
            u = pd.Series(-ws * np.sin((math.pi/180) * wd))
            v = pd.Series(-ws * np.cos((math.pi/180) * wd))
            ws_u.append(u)
            ws_v.append(v)

        u_final = np.sum([u / d for u, d in zip(ws_u, dd)], axis=0) / np.sum([1 / d for d in dd])
        v_final = np.sum([v / d for v, d in zip(ws_v, dd)], axis=0) / np.sum([1 / d for d in dd])

        ws_final = transformation._convert_to_ws(u_final, v_final)
        wd_final = transformation._convert_to_degrees(u_final, v_final)
        # Convert wd to meteorological degrees
        wd_final = (270 - wd_final) % 360

        dt = dt.reset_index(drop=True)
        dt.name = "datetime"
        df = pd.DataFrame(dt)
        df["ws"] = ws_final
        df["wd"] = wd_final
        return df
    elif height < 10:
        lower_height = 10
        upper_height = 20
    elif height > 1000:
        lower_height = 500
        upper_height = 1000
    else:
        lower_height = wtk_heights[wtk_heights < height].max()
        upper_height = wtk_heights[wtk_heights > height].min()

    dt = _get_dt(myr, start_time, end_time, time_stride)
    dd, ii = myr.tree.query((lat, lon), 4)

    ws_u_lower = list()
    ws_u_upper = list()
    ws_v_lower = list()
    ws_v_upper = list()
    for idx in ii:
        ws_lower = myr["windspeed_%sm" % lower_height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]
        wd_lower = myr["winddirection_%sm" % lower_height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]
        ws_upper = myr["windspeed_%sm" % upper_height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]
        wd_upper = myr["winddirection_%sm" % upper_height, dt.index[0]:dt.index[-1] + 1:time_stride, idx]

        # Convert wd to mathematical degrees
        wd_lower = (270 - wd_lower) % 360
        wd_upper = (270 - wd_upper) % 360

        u_lower = pd.Series(-ws_lower * np.sin((math.pi/180) * wd_lower))
        u_upper = pd.Series(-ws_upper * np.sin((math.pi/180) * wd_upper))
        v_lower = pd.Series(-ws_lower * np.cos((math.pi/180) * wd_lower))
        v_upper = pd.Series(-ws_upper * np.cos((math.pi/180) * wd_upper))

        ws_u_lower.append(u_lower)
        ws_u_upper.append(u_upper)
        ws_v_lower.append(v_lower)
        ws_v_upper.append(v_upper)

    u_lower = np.sum([u / d for u, d in zip(ws_u_lower, dd)], axis=0) / np.sum([1 / d for d in dd])
    u_upper = np.sum([u / d for u, d in zip(ws_u_upper, dd)], axis=0) / np.sum([1 / d for d in dd])
    v_lower = np.sum([v / d for v, d in zip(ws_v_lower, dd)], axis=0) / np.sum([1 / d for d in dd])
    v_upper = np.sum([v / d for v, d in zip(ws_v_upper, dd)], axis=0) / np.sum([1 / d for d in dd])

    u_final = power_law(u_lower, u_upper, lower_height, upper_height, height)
    v_final = power_law(v_lower, v_upper, lower_height, upper_height, height)

    ws_final = transformation._convert_to_ws(u_final, v_final)
    wd_final = transformation._convert_to_degrees(u_final, v_final)
    # Convert wd to meteorological degrees
    wd_final = (270 - wd_final) % 360

    dt = dt.reset_index(drop=True)

    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_final
    df["wd"] = wd_final

    return df



def _get_dt(myr, start_time=None, end_time=None, time_stride=None):
    dt = myr.time_index
    dt = pd.DataFrame({"datetime": dt[:]}, index=range(0,dt.shape[0]))
    dt = dt["datetime"]
    if (start_time is not None) and (end_time is not None) and (time_stride is not None):
        dt=dt.loc[(dt >= start_time) & (dt <= end_time)].iloc[::time_stride]
    return dt

def _getDateTime(f):
    """ Retrieves and parses date and time from data returning dt["datetime"] """
    dt = f["datetime"]
    dt = pd.DataFrame({"datetime": dt[:]},index=range(0,dt.shape[0]))
    dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)
    dt["datetime"] = pd.to_datetime(dt['datetime'])
    return dt["datetime"]

def _indicesForCoord(f, lat, lon):
    """
    Determines the WTK grid box in which the latitude and longitude of the turbine candidate point lies in. Assumes grid box dimensions of 2000 meters by 2000 meters. Uses the lcc projection to convert from degrees to meters.
    Inputs: read in file, latitude, longitude
    Outputs: List of grid index tuples for WTK bounding box starting from bottom left and moving clockwise. List of distances from point to each bounding box corner, ordered from bottom left corner moving clockwise. List of meters coordinates of the box starting from bottom left and moving clockwise. Meter projection of longitude. Meter projection of latitude.
    """
    #Clockwise starting bottom left
    dset_coords = f['coordinates']
    projstring = """+proj=lcc +lat_1=30 +lat_2=60
                    +lat_0=38.47240422490422 +lon_0=-96.0
                    +x_0=0 +y_0=0 +ellps=sphere
                    +units=m +no_defs """
    projectLcc = Proj(projstring)
    #WTK coords are in lat lon -> origin_ll lon lat
    origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
    origin = projectLcc(*origin_ll)
    coords = (lon,lat)
    coords = projectLcc(*coords) #meters transformation
    delta = np.subtract(coords, origin) #lon lat
    tuples = []

    tuples.append(tuple(reversed([int(math.floor(x/2000)) for x in delta]))) #(min, min) #bottom left
    tuples.append(tuple(reversed([int(math.floor(delta[0]/2000)), int(math.floor(delta[1]/2000)) + 1]))) #(max, min) top left
    tuples.append(tuple(reversed([int(math.floor(x/2000)) + 1 for x in delta]))) #(max, max) #top right
    tuples.append(tuple(reversed([int(math.floor(delta[0]/2000)) + 1, int(math.floor(delta[1]/2000))]))) #(min, max) #bottom right

    bottom_left = np.array([int(math.floor(x/2000)) for x in delta]) #bottom left, lon, lat
    bottom_left_delta = np.subtract(delta, bottom_left * 2000) #bottom left
    bottom_left_dist = math.sqrt(np.sum(np.inner(bottom_left_delta, bottom_left_delta))) #bottom left
    top_left_delta = np.subtract(delta, ((bottom_left + [0, 1]) * 2000)) #top left
    top_left_dist = math.sqrt(np.sum(np.inner(top_left_delta, top_left_delta))) #top left
    top_right_delta = np.subtract(delta, (bottom_left + [1, 1]) * 2000) #top right
    top_right_dist = math.sqrt(np.sum(np.inner(top_right_delta, top_right_delta))) #top right
    bottom_right_delta = np.subtract(delta, (bottom_left + [1, 0]) * 2000) #bottom right
    bottom_right_dist = math.sqrt(np.sum(np.inner(bottom_right_delta, bottom_right_delta))) #bottom right
    dist = [bottom_left_dist, top_left_dist, top_right_dist, bottom_right_dist]
    try:
        grid_points = [projectLcc(*(reversed((dset_coords[bottom_left[1]][bottom_left[0]])))),
                   projectLcc(*(reversed((dset_coords[bottom_left[1] + 1][bottom_left[0]])))),
                   projectLcc(*(reversed((dset_coords[bottom_left[1] + 1][bottom_left[0] + 1])))),
                   projectLcc(*(reversed((dset_coords[bottom_left[1]][bottom_left[0] + 1]))))]
    except:
        grid_points = [-1, -1, -1, -1]
    return tuples, dist, grid_points, coords[0], coords[1]


def _interpolate_spatially_row(row, dist, grid_points, x, y, method):
    """ This function provides per-row spatial interpolatoin using
    nearest, linear, cubic, and IDW (inverse-distance weighting) methods.
    It is conveninet to use this function with df.apply().
    """
    # Important: order in dist array should exactly match columns in dataframe where row is coming from
    if method in ["nearest", "linear", "cubic"]:
        result = griddata(grid_points, row.values,
                          ([x], [y]), method=method)[0]
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    elif method == "IDW":
        numerator = 0
        denominator = 0
        for idx in range(len(row.values)):
            w = 1.0 / dist[idx]
            numerator += w * row.values[idx]
            denominator += w
        result = numerator/denominator
    return result

def _interpolate_vertically(lat, lon, wind_below, wind_above, height, desired_point, method, height_below = 40, height_above = 60):
    """
    Vertically interpolates variables based on predicted WTK values of heights 40 and 60 meters.
    """
    p_below = points.XYZPoint(lat, lon, height_below, 'model',
                          timeseries=[timeseries.timeseries(wind_below, var="ws")])
    p_above = points.XYZPoint(lat, lon, height_above, 'model',
                          timeseries=[timeseries.timeseries(wind_above, var="ws")])

    xyz_points = [p_below, p_above]
    xy_point = points.XYPoint.from_xyz_points(xyz_points)

    #Interpolation
    # supported vertical_interpolation_techniques:
    # nn, polynomial, stability_adjusted_log_law, neutral_log_law, stability_adjusted_power_law, neutral_power_law

    # neutral_power_law seems to be acting weird (results seem to be too close to "below" or "above" values...

    # Not currently working: stability_adjusted_power_law

    # Working (should be enough for now, to continue development)" nn, polynomial

    vi = interpolation.interpolation(
        desired_point,
        xy_point,
        vertically_interpolate=True,
        spatially_interpolate=False,
        vertical_interpolation_techniques=method)
    vi.interpolate()

    interpolated = vi._model_transformed[0]._xyz_points._time_series[0]._timeseries
    return interpolated
