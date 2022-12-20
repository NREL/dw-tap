from dw_tap.data_processing import geojson_toCoordinate
from dw_tap.data_processing import prepare_data
#from dw_tap.loadMLmodel import loadMLmodel
from dw_tap.LOMvectorized import loadMLmodel

import numpy as np
import time
import pandas as pd
import pyproj

def run_lom(df, df_places, xy_turbine, z_turbine):
    print("run_lom: starting")
    footprint_size = 1000
    dates, ws, theta = df["datetime"], df["ws"], df["wd"]
    x1_turbine, y1_turbine = xy_turbine[0][0], xy_turbine[0][1]
    x1_turbine, y1_turbine = _LatLon_To_XY(y1_turbine, x1_turbine)
    minx = x1_turbine - footprint_size 
    maxx = x1_turbine + footprint_size
    miny = y1_turbine - footprint_size
    maxy = y1_turbine + footprint_size
    
    t0 = time.time()
    model = loadMLmodel()
    print("run_lom: loaded model")
    
    trees = False #True #False #True --> use trees #False --> don't use trees
    porosity = 0.0  
    xy,H, eps=geojson_toCoordinate(df_places,minx,maxx,miny,maxy,trees, porosity)
    eps=np.array(eps) #make an array of porosities
    print("run_lom: after geojson_toCoordinate")

    #centroid x, centroid y, transformed XYp[x,y], rotated XYr[theta][x,y], L[theta], W[theta],XYti
    xc,yc,xyp,xyr,L,W, xyt =prepare_data(xy,xy_turbine, theta)
    print("run_lom: after prepare_data")
    #xc not used, yc not used, xyp not used, xyr not used

    # beginning of vectorized version
    
    t0 = time.time()

    plot_test_data = np.zeros((len(L[0])*len(L)*len(xy_turbine),6))
    wss=np.zeros(len(L[0])*len(L)*len(xy_turbine))
    kk=0
    for i in range(len(xy_turbine)): #loop over turbines number#
        for j in range (len(L)):    #loop over building number#
            for k in range (len(L[0])): #loop over theta

                plot_test_data[kk,0] = H[j]/H[j]   #H changes between objects
                plot_test_data[kk,1] = W[j][k]/H[j]   #W alters with theta
                plot_test_data[kk,2] = L[j][k]/H[j]   #L alters with theta

                plot_test_data[kk,3] = abs(xyt[j][i][k,1]/H[j]) #s_turbine: alters with theta stramwise direction
                plot_test_data[kk,4] = (xyt[j][i][k,0])/H[j]   #w_turbine: alters with theta -spanwise direction
                plot_test_data[kk,5] = z_turbine/H[j]  #z[:]: constant
                wss[kk]=ws[k]
                #plot_test_data[kk,6] = 0.0  #z[:]: constant

                kk=kk+1

    t1 = time.time()
    total = t1-t0

    print('time to fill arrays :', total, ' sec')

    #if model == "LOM":
    outputs_0  = model.make_predictions(plot_test_data) 
    print("outputs_0", outputs_0)
    #if model == "ML":
    #    outputs_0  = LOMML.make_predictions(plot_test_data)            

    f=np.zeros(len(outputs_0))
    f=(outputs_0[:,0])*(wss)*np.power((plot_test_data[:,0])/z_turbine,0.143)#*(1.-eps[j])
    out2=f.reshape(len(L),len(L[0])).T

    #fnl=np.zeros(ws[:])
    fl =out2.sum(axis=1)
    fnl1=out2[:]*out2[:]
    fnl =fnl1.sum(axis=1)

    upnl = ws[:]-fnl[:]
    upl = ws[:]-fl[:]

    #fnlsum[i,k] =np.sqrt(np.sum(f[i,:,k]*f[i,:,k]))            

    t1 = time.time()
    total = t1-t0

    print('computation time :', total, ' sec')

    predictions_df = pd.DataFrame({'timestamp': dates, 'linear':upl, 'nonlinear':upnl})
    predictions_df['wtk'] = ws 
    
    # end of vectorized version
    
    print('LOM time :', np.round(total/60,2), ' min')
    
    #return predictions_df
    # Clean up ANL's output
    return predictions_df.rename(columns={"wtk": "ws", "nonlinear": "ws-adjusted"}).drop(columns=["linear"])
    
def _LatLon_To_XY(Lat,Lon):
    """
    Input: Lat, Lon coordinates in degrees. 
    _LatLon_To_XY uses the albers projection to transform the lat lon coordinates in degrees to meters. 
    This is an internal function called by get_candidate.
    Output: Meter representation of coordinates. 
    """
    P = pyproj.Proj("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
    return P(Lon, Lat) #returned x, y note: lon has to come first here when calling