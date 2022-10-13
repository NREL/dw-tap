import pandas as pd
import numpy as np

def estimate_power_output(df, temp, pres, ws_column="ws-adjusted"): 
    """
    Inputs: dataframe, temperature series and pressure series
    Outputs: total kw predicted over time period, instances with wind speed above possible generation, instances with wind speed below possible generation, lists of wind speeds above and below those marks
    """
    air_density = (pres) / (287.05 * temp)
    df["nonlinear"] = (df[ws_column] * ((air_density/1.225)**(1/3)))
    kw = Bergey10.windspeed_to_kw(df)
    above_curve_counter = Bergey10.above_curve_counter
    below_curve_counter = Bergey10.below_curve_counter
    above_curve_list = Bergey10.above_curve_list
    below_curve_list = Bergey10.below_curve_list
    return kw, above_curve_counter, below_curve_counter, above_curve_list, below_curve_list

class Bergey10(object):
    
    # Load data and minimal preprocessing
    raw_data = pd.read_excel("../bergey/bergey_excel10_powercurve.xlsx")
    raw_data.rename(columns={"Wind Speed (m/s)": "ws", "Turbine Output": "kw"}, inplace=True)
    
    # Create vectors for interpolation
    interp_x = raw_data.ws
    interp_y = raw_data.kw
    
    # Counters for cases outside of the real curve
    below_curve_counter = 0
    above_curve_counter = 0
    # Keeping windspeeds that are higher than what is in the curve
    above_curve_list = []
    below_curve_list = []
    
    max_ws = max(raw_data.ws)
    
    @classmethod
    def windspeed_to_kw(cls, df):
        """ Converts wind speed to kw """
        kw = pd.Series(np.interp(df["nonlinear"], cls.interp_x, cls.interp_y))
        ws = df["nonlinear"]
        for i in range(len(kw)):
            if kw.loc[i] <= 0: 
                cls.below_curve_counter += 1
                cls.below_curve_list.append(tuple((df["timestamp"][i], kw[i])))
            
            if ws.loc[i] > cls.max_ws:
                cls.above_curve_counter += 1
                cls.above_curve_list.append(tuple((df["timestamp"][i], ws[i])))
                kw.loc[i] = 0
        
        return kw
    
    @classmethod
    def reset_counters(cls):
        """ Sets counters and lists back to 0 """
        cls.below_curve_counter = 0
        cls.above_curve_counter = 0
        cls.above_curve_list = []
        cls.below_curve_list = []