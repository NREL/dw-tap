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

def get_wtk_data_nn(f, lat, lon, height, 
               start_time=None, end_time=None, time_stride=None): 
    dset_coords = f['coordinates']
    projstring = """+proj=lcc +lat_1=30 +lat_2=60 
                    +lat_0=38.47240422490422 +lon_0=-96.0 
                    +x_0=0 +y_0=0 +ellps=sphere 
                    +units=m +no_defs """
    projectLcc = Proj(projstring)
    origin_ll = reversed(dset_coords[0][0])  # Grab origin directly from database
    origin = projectLcc(*origin_ll)
    coords = (lon,lat)
    coords = projectLcc(*coords)
    delta = np.subtract(coords, origin)
    ij = [int(round(x/2000)) for x in delta]
    nn_index = tuple(reversed(ij))

    # Skip this site if it is not within the bounds of the WTK data
    # print(dset_coords.shape)
    # print(nn_index)
    if nn_index[0] < 0 or nn_index[1] < 0:
        return None
    if nn_index[0] >= dset_coords.shape[0] or nn_index[1] >= dset_coords.shape[1]:
        return None
        
    dt = _getDateTime(f)

    # dt = f["datetime"]
    # dt = pd.DataFrame({"datetime": dt[:]},index=range(0,dt.shape[0]))
    # dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)
    # dt["datetime"] = pd.to_datetime(dt['datetime'])
    # return dt["datetime"]
    
    # Determine the model data to retrieve based on input parameters
    if (start_time is not None) and (end_time is not None) and (time_stride is not None):
        # All three are specified
        #dt=dt.iloc[start_time_idx:end_time_idx+1:time_stride].reset_index(drop=True)
        dt=dt.loc[(dt >= start_time) & (dt <= end_time)].iloc[::time_stride]

    wtk_heights = np.array([10, 40, 60, 80, 100, 120, 140, 160, 200])
    if height in wtk_heights:
        ws = f['windspeed_%sm' % height][dt.index[0]:dt.index[-1]:time_stride, nn_index[0], nn_index[1]]
        wd = f['winddirection_%sm' % height][dt.index[0]:dt.index[-1]:time_stride, nn_index[0], nn_index[1]]
        dt = dt.reset_index(drop=True)
    
        dt.name = "datetime"
        df = pd.DataFrame(dt)
        df["ws"] = ws
        df["wd"] = wd
        
        return df
    elif height < 40: 
        lower_height = 40
        upper_height = 60
    elif height > 200:
        lower_height = 160
        upper_height = 200
    else: 
        lower_height = wtk_heights[wtk_heights < height].max()
        upper_height = wtk_heights[wtk_heights > height].min()
        
    # ws_lower = f['windspeed_%sm' % lower_height][dt.index[0]:dt.index[-1] + 1:time_stride, nn_index[0], nn_index[1]]
    # print(dt.index)
    # print(nn_index)
    # print(lower_height, upper_height)
    ws_lower = f['windspeed_%sm' % lower_height][dt.index[0]:dt.index[-1]:time_stride, nn_index[0], nn_index[1]]
    ws_upper = f['windspeed_%sm' % upper_height][dt.index[0]:dt.index[-1]:time_stride, nn_index[0], nn_index[1]]
    wd_lower = f['winddirection_%sm' % lower_height][dt.index[0]:dt.index[-1]:time_stride, nn_index[0], nn_index[1]]
    wd_upper = f['winddirection_%sm' % upper_height][dt.index[0]:dt.index[-1]:time_stride, nn_index[0], nn_index[1]]

    # wd_lower = pd.Series(wd_lower).apply(transformation._convert_to_met_deg)
    # wd_upper = pd.Series(wd_upper).apply(transformation._convert_to_met_deg)
    wd_lower = (270 - wd_lower) % 360
    wd_upper = (270 - wd_upper) % 360

    u_lower = pd.Series(-ws_lower * np.sin((math.pi/180) * wd_lower))
    u_upper = pd.Series(-ws_upper * np.sin((math.pi/180) * wd_upper))
    v_lower = pd.Series(-ws_lower * np.cos((math.pi/180) * wd_lower))
    v_upper = pd.Series(-ws_upper * np.cos((math.pi/180) * wd_upper))

    desired_point = points.XYZPoint(lat, lon, height, 'desired')
    u_final = _interpolate_vertically(lat, lon, 
                                      u_lower, u_upper, 
                                      height, desired_point, 
                                      "polynomial")
    v_final = _interpolate_vertically(lat, lon, 
                                      v_lower, v_upper, 
                                      height, desired_point, 
                                      "polynomial")

    ws_result = transformation._convert_to_ws(u_final, v_final) 
    wd_result = transformation._convert_to_degrees(u_final, v_final)
    # wd_result = wd_result.apply(transformation._convert_to_math_deg, args=())
    wd_result = (270 - wd_result) % 360

    # ws = f['windspeed_%sm' % height][dt.index[0]:dt.index[-1] + 1:time_stride, nn_index[0], nn_index[1]]
    # wd = f['winddirection_%sm' % height][dt.index[0]:dt.index[-1] + 1:time_stride, nn_index[0], nn_index[1]]
    dt = dt.reset_index(drop=True)

    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_result
    df["wd"] = wd_result
    
    return df

def get_data_wtk_led_on_eagle(myr, 
                             lat, lon, height, 
                             method='IDW', 
                             power_estimate=False,
                             start_time_idx=None, 
                             end_time_idx=None, 
                             time_stride=None): 
    """
    Horizontally and vertically interpolates wind speed, wind direction, and potentially temperature and pressure. Horizontal interpolation for wind speed and direction uses inverse distance weight. Horizontal interpolation for temperature and pressure uses nearest neighbor. Vertical interpolation uses linear method for all variables. 
    Input: read in file, latitude, longitude, height of potential turbine/candidate turbine, horizontal interpolation method, boolean indicating whether temperature and pressure data will be needed for power estimate. 
    Output: (1) If power_estimate is true, returns are wind speed, wind direction, datetime, temperature and pressure. (2) If power_estimate is false, returns are wind speed, wind direction, and datetime. All values horizontally and vertically interpolated. 
    """
    dt = myr.time_index
    dt = pd.DataFrame({"datetime": dt[:]}, index=range(0,dt.shape[0]))
    dt = dt["datetime"]
    
    if (start_time_idx is not None) and (end_time_idx is not None) and (time_stride is not None):
        # All three are specified
        dt=dt.iloc[start_time_idx:end_time_idx+1:time_stride].reset_index(drop=True)
    else: 
        start_time_idx = 0
        end_time_idx = len(dt)
        time_stride=1
    #print("Selected time index range::\n", dt)   
        
    desired_point = points.XYZPoint(lat, lon, height, 'desired')    
        
    dd, ii = myr.tree.query((lat, lon), 4)
    #print("Distances and indices:\n", dd,ii) 
    # Example output: [0.01282314 0.0143017  0.01750789 0.01969273] [2663301 2665250 2663302 2665251]
    # Note that ii indices aren't contiguous
    
    # Fetching wind speed
    selected_ds = "windspeed_40m"
    ws1 = pd.DataFrame(index = dt.index)
    for idx in range(len(ii)):
        one_series = myr[selected_ds, start_time_idx:end_time_idx+1:time_stride, ii[idx]]
        ws1[str(idx+1)] = one_series
    #print(ws1)
    
    selected_ds = "windspeed_60m"
    ws2 = pd.DataFrame(index = dt.index)
    for idx in range(len(ii)):
        one_series = myr[selected_ds, start_time_idx:end_time_idx+1:time_stride, ii[idx]]
        ws2[str(idx+1)] = one_series
    #print(ws2)
    
    # Fetching wind direction
    selected_ds = "winddirection_40m"
    wd1 = pd.DataFrame(index = dt.index)
    for idx in range(len(ii)):
        one_series = myr[selected_ds, start_time_idx:end_time_idx+1:time_stride, ii[idx]]
        wd1[str(idx+1)] = one_series
    #print(wd1)
    
    selected_ds = "winddirection_60m"
    wd2 = pd.DataFrame(index = dt.index)
    for idx in range(len(ii)):
        one_series = myr[selected_ds, start_time_idx:end_time_idx+1:time_stride, ii[idx]]
        wd2[str(idx+1)] = one_series
    #print(wd2)
    
    #Spatial for vectors (horizontal and vertical)
    dist = dd
    
    #U vector
    wd1 = wd1.apply(transformation._convert_to_met_deg, args=(), axis = 1)
    wd2 = wd2.apply(transformation._convert_to_met_deg, args=(), axis = 1)
    
    u, u1 = transformation._convert_to_vector_u(wd1, wd2, ws1, ws2)
    
    #u["spatially_interpolated"] = u.apply(_interpolate_spatially_row, 
    #                                      args=(dist, grid_points, x, y, method), axis=1)
    # 
    # 'IDW' doesn't use gridpoints - going for a quick implementation for IDW only; x and y aren't needed either
    u["spatially_interpolated"] = u.apply(_interpolate_spatially_row, 
                                          args=(dist, [], [], [], 'IDW'), axis=1)
    
    u1["spatially_interpolated"] = u1.apply(_interpolate_spatially_row, 
                                           args=(dist, [], [], [], 'IDW'), axis=1)
    u = pd.Series(u["spatially_interpolated"], name='wd')
    u1 = pd.Series(u1["spatially_interpolated"], name='wd')
    
    u_final = _interpolate_vertically(lat, lon, u, u1, height, desired_point, "polynomial")
    #print("u_final:\n", u_final)
    
    v, v1 = transformation._convert_to_vector_v(wd1, wd2, ws1, ws2)
    v["spatially_interpolated"] = v.apply(_interpolate_spatially_row, 
                                          args=(dist, [], [], [], 'IDW'), axis=1)
    v1["spatially_interpolated"] = v1.apply(_interpolate_spatially_row, 
                                            args=(dist, [], [], [], 'IDW'), axis=1)
    v = pd.Series(v["spatially_interpolated"], name='wd')
    v1 = pd.Series(v1["spatially_interpolated"], name='wd')
    v_final = _interpolate_vertically(lat, lon, v, v1, height, desired_point, "polynomial")
    #print("v_final:\n", v_final)
    
    ws_result = transformation._convert_to_ws(v_final, u_final) 
    wd_result = transformation._convert_to_degrees(v_final, u_final)
    #print("ws_result:\n", ws_result)
    #print("wd_result:\n", wd_result)
    
    wd_result = wd_result.apply(transformation._convert_to_math_deg, args=())
    
    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_result
    df["wd"] = wd_result
    
    if power_estimate == True: 
         print("Warning: WTK-LED does not include pressure and temperature timeseries. Power estimation is currently unavailable.")
    
    return df 


def get_data_wtk_led_nn(myr_pathstr,
                        lat, lon, height, 
                        start_time=None, end_time=None, time_stride=None): 
    """
    Vertically interpolates wind speed and wind direction. 
    Vertical interpolation uses linear method for all variables. 
    Input: path string to get read in file, latitude, longitude, height of potential turbine/candidate turbine
    Output: Pandas DataFrame of wind speed, wind direction, and datetime. 
    """
    dt = myr.time_index
    dt = pd.DataFrame({"datetime": dt[:]}, index=range(0,dt.shape[0]))
    dt = dt["datetime"]
    
    if (start_time is not None) and (end_time is not None) and (time_stride is not None):
        dt=dt.loc[(dt >= start_time) & (dt <= end_time)].iloc[::time_stride]
        
    desired_point = points.XYZPoint(lat, lon, height, 'desired')
        
    dd, ii = myr.tree.query((lat, lon), 1) # This implementation does not use dd
    
    wtk_heights = np.array([10, 40, 60, 80, 100, 120, 140, 160, 200])
    if height in wtk_heights:
        if 'ANL_4km_north_america' in myr_pathstr:
            myr = MultiYearWindX(myr_pathstr, hsds=False)
        else:
            myr = MultiYearWindX(myr_pathstr + '_%sm.h5' % height, hsds=False)
        ws = myr['windspeed_%sm' % height, dt.index[0]:dt.index[-1] + 1:time_stride, ii]
        wd = myr['winddirection_%sm' % height, dt.index[0]:dt.index[-1] + 1:time_stride, ii]
        dt = dt.reset_index(drop=True)
        dt.name = "datetime"
        df = pd.DataFrame(dt)
        df["ws"] = ws
        df["wd"] = wd
        return df
    elif height < 40: 
        lower_height = 40
        upper_height = 60
    elif height > 200:
        lower_height = 160
        upper_height = 200
    else: 
        lower_height = wtk_heights[wtk_heights < height].max()
        upper_height = wtk_heights[wtk_heights > height].min()

    if 'ANL_4km_north_america' in myr_pathstr:
        myr = MultiYearWindX(myr_pathstr, hsds=False)

        # Fetching wind speed & direction
        ws_lower = myr["windspeed_%sm" % lower_height, dt.index[0]:dt.index[-1]:time_stride, ii] 
        wd_lower = myr["winddirection_%sm" % lower_height, dt.index[0]:dt.index[-1]:time_stride, ii]
        ws_upper = myr["windspeed_%sm" % upper_height, dt.index[0]:dt.index[-1]:time_stride, ii]
        wd_upper = myr["winddirection_%sm" % upper_height, dt.index[0]:dt.index[-1]:time_stride, ii]
    else:
        myr_lower = MultiYearWindX(myr_pathstr + '_%sm.h5' % lower_height, hsds=False)
        myr_upper = MultiYearWindX(myr_pathstr + '_%sm.h5' % upper_height, hsds=False)
        
        # Fetching wind speed & direction
        ws_lower = myr_lower["windspeed_%sm" % lower_height, dt.index[0]:dt.index[-1]:time_stride, ii] 
        wd_lower = myr_lower["winddirection_%sm" % lower_height, dt.index[0]:dt.index[-1]:time_stride, ii]
        ws_upper = myr_upper["windspeed_%sm" % upper_height, dt.index[0]:dt.index[-1]:time_stride, ii]
        wd_upper = myr_upper["winddirection_%sm" % upper_height, dt.index[0]:dt.index[-1]:time_stride, ii]

    # Convert wd to mathematical degrees
    wd_lower = (270 - wd_lower) % 360
    wd_upper = (270 - wd_upper) % 360

    u_lower = pd.Series(-ws_lower * np.sin((math.pi/180) * wd_lower))
    u_upper = pd.Series(-ws_upper * np.sin((math.pi/180) * wd_upper))
    v_lower = pd.Series(-ws_lower * np.cos((math.pi/180) * wd_lower))
    v_upper = pd.Series(-ws_upper * np.cos((math.pi/180) * wd_upper))
    
    desired_point = points.XYZPoint(lat, lon, height, 'desired')
    u_final = _interpolate_vertically(lat, lon, 
                                      u_lower, u_upper, 
                                      height, desired_point, 
                                      "polynomial")
    v_final = _interpolate_vertically(lat, lon, 
                                      v_lower, v_upper, 
                                      height, desired_point, 
                                      "polynomial")

    ws_result = transformation._convert_to_ws(u_final, v_final) 
    wd_result = transformation._convert_to_degrees(u_final, v_final)
    # Convert wd to meteorological degrees
    wd_result = (270 - wd_result) % 360

    dt = dt.reset_index(drop=True)
    
    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_result
    df["wd"] = wd_result
    
    return df 
    

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

