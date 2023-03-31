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

def getData(f, lat, lon, height, method='IDW', 
           power_estimate=False,
           inverse_monin_obukhov_length = False, 
           start_time_idx=None, end_time_idx=None, time_stride=None): 
    """
    Horizontally and vertically interpolates wind speed, wind direction, and 
    potentially temperature and pressure. Horizontal interpolation for wind 
    speed and direction uses inverse distance weight. 
    
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
    dt = _getDateTime(f)
    
    if (start_time_idx is not None) and (end_time_idx is not None) and (time_stride is not None):
        # All three are specified
        dt=dt.iloc[start_time_idx:end_time_idx+1:time_stride].reset_index(drop=True)
    else: 
        start_time_idx = 0
        end_time_idx = len(dt)
        time_stride=1
        
    point_lat_lon = (lat, lon)
    point_idx, dist, grid_points, x, y = _indicesForCoord(f, point_lat_lon[0], point_lat_lon[1])
    desired_point = points.XYZPoint(lat, lon, height, 'desired')
    minx = min(point_idx,key=itemgetter(0))[0]
    maxx = max(point_idx,key=itemgetter(0))[0]
    miny = min(point_idx,key=itemgetter(1))[1] 
    maxy = max(point_idx,key=itemgetter(1))[1]
    
    #Wind Speed Fetching
    dset = f['windspeed_40m']
    ws = dset[start_time_idx:end_time_idx+1:time_stride, 
              minx:maxx+1, miny:maxy+1]
    ws = pd.DataFrame({'1': ws[:, 0, 0], '2': ws[:, 1, 0], '3': ws[:, 1, 1], '4': ws[:, 0, 1]})
    dset = f['windspeed_60m']
    ws1 = dset[start_time_idx:end_time_idx+1:time_stride,
               minx:maxx+1, miny:maxy+1]
    ws1 = pd.DataFrame({'1': ws1[:, 0, 0], '2': ws1[:, 1, 0], '3': ws1[:, 1, 1], '4': ws1[:, 0, 1]})
    
    #Wind Direction Fetching
    dset = f['winddirection_40m']
    wd = dset[start_time_idx:end_time_idx+1:time_stride,
              minx:maxx+1, miny:maxy+1]
    wd = pd.DataFrame({'1': wd[:, 0, 0], '2': wd[:, 1, 0], '3': wd[:, 1, 1], '4': wd[:, 0, 1]})
    dset = f['winddirection_60m']
    wd1 = dset[start_time_idx:end_time_idx+1:time_stride,
               minx:maxx+1, miny:maxy+1]
    wd1 = pd.DataFrame({'1': wd1[:, 0, 0], '2': wd1[:, 1, 0], '3': wd1[:, 1, 1], '4': wd1[:, 0, 1]})
    
    #Spatial for vectors (horizontal and vertical)
    #U vector
    wd = wd.apply(transformation._convert_to_met_deg, args=(), axis = 1)
    wd1 = wd1.apply(transformation._convert_to_met_deg, args=(), axis = 1)
    u, u1 = transformation._convert_to_vector_u(wd, wd1, ws, ws1)
    u["spatially_interpolated"] = u.apply(_interpolate_spatially_row, args=(dist,grid_points,x,y,method), axis=1)
    u1["spatially_interpolated"] = u1.apply(_interpolate_spatially_row, args=(dist,grid_points,x,y,method), axis=1)
    u = pd.Series(u["spatially_interpolated"], name='wd')
    u1 = pd.Series(u1["spatially_interpolated"], name='wd')
    u_final = _interpolate_vertically(lat, lon, u, u1, height, desired_point, "polynomial")

    #V vector
    wd = wd.apply(transformation._convert_to_met_deg, args=(), axis = 1)
    wd1 = wd1.apply(transformation._convert_to_met_deg, args=(), axis = 1)
    v, v1 = transformation._convert_to_vector_u(wd, wd1, ws, ws1)
    v["spatially_interpolated"] = v.apply(_interpolate_spatially_row, args=(dist,grid_points,x,y,method), axis=1)
    v1["spatially_interpolated"] = v1.apply(_interpolate_spatially_row, args=(dist,grid_points,x,y,method), axis=1)
    v = pd.Series(v["spatially_interpolated"], name='wd')
    v1 = pd.Series(v1["spatially_interpolated"], name='wd')
    v_final = _interpolate_vertically(lat, lon, v, v1, height, desired_point, "polynomial")
    
    ws_result = transformation._convert_to_ws(v_final, u_final) 
    wd_result = transformation._convert_to_degrees(u_final, v_final)
    
    wd_result = wd_result.apply(transformation._convert_to_math_deg, args=())
    
    dt.name = "datetime"
    df = pd.DataFrame(dt)
    df["ws"] = ws_result
    df["wd"] = wd_result
    
    if power_estimate: 
        dtemp = f['temperature_40m']
        dtemp = dtemp[start_time_idx:end_time_idx+1:time_stride,
                      minx:maxx+1, miny:maxy+1]
        dtemp = pd.DataFrame({'1': dtemp[:,0,0], '2': dtemp[:,1,0], '3': dtemp[:,1,1], '4': dtemp[:,0,1]})
        dtemp1 = f['temperature_60m']
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
    
    grid_points = [projectLcc(*(reversed((dset_coords[bottom_left[1]][bottom_left[0]])))), 
                   projectLcc(*(reversed((dset_coords[bottom_left[1] + 1][bottom_left[0]])))), 
                   projectLcc(*(reversed((dset_coords[bottom_left[1] + 1][bottom_left[0] + 1])))),
                   projectLcc(*(reversed((dset_coords[bottom_left[1]][bottom_left[0] + 1]))))]
    
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

