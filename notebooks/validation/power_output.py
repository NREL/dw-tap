import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

def estimate_power_output(df, temp, pres, ws_column="ws-adjusted"): 
    """
    Inputs: dataframe, temperature series and pressure series
    Outputs: total kw predicted over time period, instances with wind speed above possible generation, instances with wind speed below possible generation, lists of wind speeds above and below those marks
    """
    df_copy = df.copy()
    
    air_density = (pres) / (287.05 * temp)
    df_copy[ws_column] = (df_copy[ws_column] * ((air_density/1.225)**(1/3)))
    kw = Bergey10.windspeed_to_kw(df_copy, ws_column)
    above_curve_counter = Bergey10.above_curve_counter
    below_curve_counter = Bergey10.below_curve_counter
    above_curve_list = Bergey10.above_curve_list
    below_curve_list = Bergey10.below_curve_list
    return kw, above_curve_counter, below_curve_counter, above_curve_list, below_curve_list

class Bergey10(object):
    
    # Load data and minimal preprocessing
    raw_data = pd.read_csv("../../bergey/bergey_excel10_powercurve.csv")
    raw_data.rename(columns={"Wind Speed (m/s)": "ws", "Turbine Output": "kw"}, inplace=True)
    
    # Create vectors for interpolation
    interp_x = raw_data.ws
    interp_y = raw_data.kw
    
    # Cubic interpolation
    powercurve_intrp = interp1d(interp_x, interp_y, kind='cubic')
    
    # Counters for cases outside of the real curve
    below_curve_counter = 0
    above_curve_counter = 0
    # Keeping windspeeds that are higher than what is in the curve
    above_curve_list = []
    below_curve_list = []
    
    max_ws = max(raw_data.ws)
    
    @classmethod
    def windspeed_to_kw(cls, df, ws_column="ws-adjusted"):
        """ Converts wind speed to kw """
        kw = cls.powercurve.intrp(df[ws_column])
        ws = df[ws_column]
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
        """ Sets counters and lists back to 0 and empty """
        cls.below_curve_counter = 0
        cls.above_curve_counter = 0
        cls.above_curve_list = []
        cls.below_curve_list = []
        
    @classmethod   
    def plot(cls):
        fig = px.line(y=cls.powercurve_intrp(cls.interp_x), x=cls.interp_x,
              labels={"x":"Windspeed (m/s)","y":"Power (kW)"})
        fig.add_trace(go.Scatter(y=cls.interp_y, x=cls.interp_x,
                    mode='markers',
                    name='Data'))
        fig.show()
        
    @classmethod
    def kw_to_windspeed(cls,df,kw_column="output_power_mean"):
        # Sampling a hundred points from the interpolated function
        # allows us to invert with an approximate accuracy of 12/100 or 0.1
        ws2 = np.linspace(0, 12, num=100)
        pc2 = cls.powercurve_intrp(ws2)
        return df[kw_column].map(lambda x: ws2[np.abs(pc2 - x).argmin()] )