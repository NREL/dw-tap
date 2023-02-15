# common.py
#
# Helper functions for the notebooks in this repo.
#
#
# Author: Caleb Phillips (caleb.phillips@nrel.gov)

import pandas as pd
import plotly.express as px
import numpy as np
import plotly 
from scipy.interpolate import interp1d
import glob

#### Powercurves and Helpers ####

# FIXME: this is the powercurve for the EAZ turbine, need to update to the models of turbines used in this validation
powercurve = np.array([0,0,0,0,0,0.3,0.8,1.2,1.6,2.4,3.6,4.5,5.5,7,
                       8.4,10.2,11.6,12,12.4,12.5,12.3,12.2,12.1,12,11.8,11.7,
                       11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5,
                       11.5,11.5,8,5,2,0,0,0,0,0,0])
windspeed = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
                      5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 
                      10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 
                      14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 
                      18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5])
powercurve_intrp_limited = interp1d(windspeed[4:20], powercurve[4:20], kind='cubic')
powercurve_intrp = interp1d(windspeed, powercurve, kind='cubic')

def windspeed_to_power(x):
    try:
        return powercurve_intrp(x)
    except ValueError:
        return None

#### Stats Helpers ####

def performance(x,y):
    rmse = np.sqrt(np.mean(np.power(x - y,2)))
    mae = np.mean(np.abs(x - y))   
    mean = np.mean(x - y)
    return {"rmse":rmse, "mae": mae, "mean": mean}



#### Plotting Helpers ####

def perf2title(x):
    return 'RMSE: {:.02f}, MAE: {:.02f}, Mean: {:.02f}'.format(x["rmse"],x["mae"],x["mean"])

def plot_scatter_and_hist(predicted,observed,predlabel="Predicted (m/s)",
                          obslabel="Observed (m/s)",axrange=[0,12]):
    fig = px.scatter(y=predicted, x=observed,
                    labels={"x":obslabel,"y":predlabel},
                    title=perf2title(performance(predicted,observed)))
    fig.update_xaxes(range=axrange)
    fig.update_yaxes(range=axrange)
    # plot must have same axis ranges or this will not be an x=y line
    fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])

    fig.show()
    
    fig = px.histogram(x=predicted-observed,
                  labels={"x":"Difference"},
                  title=perf2title(performance(predicted,observed)))
    fig.add_vline(x=0)
    fig.show()

def plot1224(df,errorcol="error"):
    fig = px.area(df,facet_col="hour",facet_row="month",x="sector",y=errorcol,
                  facet_row_spacing=0.003, # default is 0.07
                  facet_col_spacing=0.001, # default is 0.03
                 labels={"sector":"",errorcol:""},title="Average Error by Hour, Month and Direction (m/s)") 
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    #fig.update_yaxes(matches=None)
    return fig.show()

def plotpolar(df,title="Average Error By Sector (m/s)",errorcol="error"):
    fig = px.bar_polar(df.groupby(["sector"]).mean().reset_index(),theta="sector",r=errorcol,
                       title=title,color=errorcol)
    return fig.show()

def plot1224heatmap(df,title="Average Error Irrespective of Direction (m/s)",errorcol="error"):
    tab = df[['hour','month',errorcol]].groupby(['hour','month']).mean().reset_index().pivot(index="month",columns="hour")
    return px.imshow(tab[errorcol],labels={"x":"Hour","y":"Month"},title=title,color_continuous_scale='RdBu')



#### Energy Calculation Helpers ####

def kwh_error_by_sector(rea_ts,iec_agg,errorcol="error"):
    rea_ts["sector"] = rea_ts["winddir"].apply(sectorize)
    sector_freq = (rea_ts.groupby("sector").count()/rea_ts.shape[0]).reset_index()[["sector","timestamp"]].rename(columns={"timestamp":"frequency"})
    sector_freq = iec_agg[["sector",errorcol]].dropna().groupby("sector").mean().reset_index().merge(sector_freq,how='outer')
    sector_freq[errorcol] = sector_freq[errorcol]*24*365*sector_freq["frequency"] # convert from kw to kwh
    return sector_freq

def kwh_error_by_hms(rea_ts,iec_agg,errorcol="error"):
    rea_ts['hour'] = rea_ts['timestamp'].astype('datetime64[ns]').dt.hour
    rea_ts['month'] = rea_ts['timestamp'].astype('datetime64[ns]').dt.month
    rea_ts["sector"] = rea_ts["winddir"].apply(sectorize)
    sector_freq = (rea_ts.groupby(["sector","hour","month"]).count()/rea_ts.shape[0]).reset_index()[["sector","hour","month","timestamp"]].rename(columns={"timestamp":"frequency"})
    sector_freq = iec_agg[["sector","month","hour",errorcol]].dropna().groupby(["sector","month","hour"]).mean().reset_index().merge(sector_freq,how="outer")
    sector_freq[errorcol] = sector_freq["error"]*24*365*sector_freq["frequency"] # convert from kw to kwh
    return sector_freq

def energy_error_stats(rea_ts,iec_agg,errorcol="error",powercol="power"):
    # Expected kwh error per year using sector-average, and hour/month/sector-average
    kwhes = kwh_error_by_sector(rea_ts,iec_agg,errorcol)
    kwhehms = kwh_error_by_hms(rea_ts,iec_agg,errorcol)
    
    # Calculate the annual total avg energy production
    dc = kwhehms.merge(iec_agg[["month","hour","sector",powercol]],how="outer")
    dc["kwh"] = dc[powercol]*24*365*dc["frequency"]
    annualkwh = dc["kwh"].sum()
    
    return { "error_kwh_per_year_by_sector": kwhes[errorcol].sum(),
             "error_kwh_per_year_by_hms": kwhehms[errorcol].sum(),
             "error_relative_by_hms": 100.0*kwhehms[errorcol].sum()/annualkwh,
             "error_relative_by_sector": 100.0*kwhes[errorcol].sum()/annualkwh
           }


#### Misc Helpers ####

def sectorize(x):
    return np.floor(x/10)*10