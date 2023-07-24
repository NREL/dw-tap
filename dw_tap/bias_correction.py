import pandas as pd
import numpy as np

class BiasCorrection(object):
    
    def __init__(self, bias_corr_data_dir="."):
        # Read in and preprocess all available observational data"
        pass
    
    def find_two_closest_bc_sites(): # "(lat,lon) or P where P is a Shapely.point and limit in kilometeres"):
        
        # return B (closest point) and T (2nd closest point, used for bc evaluation) and distance from P to B
        pass
    
    def fetch_closest_model_data(model="WTK|ERA5"):
        # This function should be reused for fetching data for B, T, and P
        # Need to decide which height this gets data for and for which timeframe; maybe have some defaults for them
        # 
        # Do we need wind direction or just wind speed?
        pass
    
    def fetch_obs_data():
        # For a given point B or T, return the available windspeed data
        pass
    
    def get_error(ts1, ts2):
        
        # Use an error metric (RMSE, MAE, or something else) to compare timeseries ts1 and ts2; either can be observational or modeling data
        # This should consider timeframes and compare only time-matching data points
        pass
    
    def bias_correct(): # "(lat,lon)", model="WTK" | "ERA5"):
        
#         P = ... # Turn (lat, lon) to P, Shapely point

#         P_ws = fetch_closest_model_data(model)
        
#         B,T,dist_to_B = find_two_closest_bc_sites:
            
#         if dist_to_B > distance_limit_km:
#             bc = 0
#             return P_ws, bc
#         else:
            
#             B_model_ws = fetch_closest_model_data(model)
#             B_obs_ws = fetch_obs_data()
                        
#             mod = sm.OLS(B_obs_ws, sm.add_constant(mdf[["ws","wd","hour","month"]]))
#             res = mod.fit()

        pass

            
        
        