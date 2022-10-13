from dw_tap.data_processing import geojson_toCoordinate
from dw_tap.data_processing import prepare_data
from dw_tap.loadMLmodel import loadMLmodel

import numpy as np
import time
import pandas as pd
import pyproj

def run_lom(df, df_places, xy_turbine, z_turbine):
    footprint_size = 1000
    dates, ws, theta = df["datetime"], df["ws"], df["wd"]
    x1_turbine, y1_turbine = xy_turbine[0][0], xy_turbine[0][1]
    x1_turbine, y1_turbine = _LatLon_To_XY(y1_turbine, x1_turbine)
    minx = x1_turbine - footprint_size 
    maxx = x1_turbine + footprint_size
    miny = y1_turbine - footprint_size
    maxy = y1_turbine + footprint_size
    
    t0 = time.time()
    data, model = loadMLmodel()
    trees = False #True #False #True --> use trees #False --> don't use trees
    porosity = 0.0  
    xy,H, eps=geojson_toCoordinate(df_places,minx,maxx,miny,maxy,trees, porosity)
    eps=np.array(eps) #make an array of porosities

    #centroid x, centroid y, transformed XYp[x,y], rotated XYr[theta][x,y], L[theta], W[theta],XYti
    xc,yc,xyp,xyr,L,W, xyt =prepare_data(xy,xy_turbine, theta)

    #xc not used, yc not used, xyp not used, xyr not used

    f=np.zeros((len(xy_turbine),len(L),len(L[0])))
    fsum=np.zeros((len(xy_turbine),len(L[0])))
    fnlsum=np.zeros((len(xy_turbine),len(L[0])))
    upl=np.zeros((len(xy_turbine),len(L[0])))
    upnl=np.zeros((len(xy_turbine),len(L[0])))
    plot_test_data = np.zeros(shape=(1,6),dtype='float32')

    # running LOM for all the obstacles, turbines and angles of attack
    for i in range(len(xy_turbine)): #loop over turbines number# #not needed?
        for k in range (len(L[0])): #loop over theta #L[0] will be array of directions
            for j in range (len(L)):    #loop over building number
                if (xyt[j][i][k,1] > 0.0): #turbine upwind of the obstacle --> f =0.0
                    f[i,j,k] = 0.
                else: #turbine in the wake of the obstacle --> f =0.0
                    plot_test_data[0,0] = H[j]/H[j]   #H changes between objects
                    plot_test_data[0,1] = W[j][k]/H[j]   #W alters with theta
                    plot_test_data[0,2] = L[j][k]/H[j]   #L alters with theta
                    plot_test_data[0,3] = abs(xyt[j][i][0,1]/H[j]) #s_turbine: alters with theta stramwise direction
                    plot_test_data[0,4] = (xyt[j][i][0,0]/H[j])   #w_turbine: alters with theta -spanwise direction
                    plot_test_data[0,5] = z_turbine/H[j]  #z[:]: constant
                    outputs_0  = model.make_predictions(plot_test_data)
                    ws = np.array(ws)
                    f[i,j,:] = outputs_0*ws[:]*np.power(H[j]/15.,0.143)*(1.-eps[j])
            fsum[i,k] =np.sum(f[i,:,k])
            fnlsum[i,k] =np.sqrt(np.sum(f[i,:,k]*f[i,:,k]))
            upnl[i,k]=ws[k]-fnlsum[i,k]
            upl[i,k]=ws[k]-fsum[i,k]
    predictions_df = pd.DataFrame({'timestamp': dates, 'linear':upl[0], 'nonlinear':upnl[0]})
    predictions_df['wtk'] = ws 
    t1 = time.time()
    total = t1-t0
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