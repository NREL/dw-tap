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
    kw = PowerCurve.windspeed_to_kw(df_copy, ws_column)
    above_curve_counter = PowerCurve.above_curve_counter
    below_curve_counter = PowerCurve.below_curve_counter
    above_curve_list = PowerCurve.above_curve_list
    below_curve_list = PowerCurve.below_curve_list
    return kw, above_curve_counter, below_curve_counter, above_curve_list, below_curve_list

class PowerCurve(object):
    
    # Load data and minimal preprocessing
    raw_data = pd.read_excel("../powercurves/bergey_excel10_powercurve.xlsx")
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
    def windspeed_to_kw(cls, df, ws_column="ws-adjusted",dt_column="timestamp",trim=True):
        """ Converts wind speed to kw """
        # by default round down/up values below or under the range of the curve
        if trim:
            ws = df[ws_column].apply(lambda x: 0 if x < 0 else x).apply(lambda x: cls.max_ws if x > cls.max_ws else x)
        else:
            ws = df[ws_column]
                
        kw = cls.powercurve_intrp(ws)    
        
        cls.below_curve_list = df[dt_column][kw < 0]
        cls.above_curve_list = df[dt_column][kw > cls.max_ws]
        cls.below_curve_counter = len(cls.below_curve_list)
        cls.above_curve_counter = len(cls.above_curve_list)
        
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