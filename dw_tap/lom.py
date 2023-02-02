from dw_tap.data_processing import geojson_toCoordinate
from dw_tap.data_processing import prepare_data
#from dw_tap.loadMLmodel import loadMLmodel
from dw_tap.LOMvectorized import loadMLmodel
from dw_tap.data_processing import _LatLon_To_XY

import numpy as np
import time
import pandas as pd
import pyproj

def run_lom(df, df_places, xy_turbine, z_turbine,
           check_distance=False):
    
    footprint_size = 1000
    dates, ws, theta = df["datetime"], df["ws"], df["wd"]
    x1_turbine, y1_turbine = xy_turbine[0][0], xy_turbine[0][1]
    x1_turbine, y1_turbine = _LatLon_To_XY(y1_turbine, x1_turbine)
    # This makes sure that coordinates in meters (not in lat,lon) are used throughout this function
    xy_turbine = [np.array([x1_turbine, y1_turbine])] 
    
    minx = x1_turbine - footprint_size 
    maxx = x1_turbine + footprint_size
    miny = y1_turbine - footprint_size
    maxy = y1_turbine + footprint_size
    
    t0 = time.time()
    
    trees = False #True #False #True --> use trees #False --> don't use trees
    porosity = 0.0  
    xy, H, eps = geojson_toCoordinate(df_places, minx, maxx, miny, maxy, trees, porosity)
    eps = np.array(eps) #make an array of porosities

    if check_distance:
        deltas_m = (np.concatenate(xy) - xy_turbine)
        min_dist_m = np.sqrt(np.min([np.dot(r, r) for r in deltas_m]))
        # 3km is a reasonable default for catching cases with turbines being far away from the buildings
        if min_dist_m > 3000:
            # ToDo: replace this with proper DEBUG message
            print("WARNING: studied point is too far buildings (min dist: %.1fm); velocity deficit=0" % min_dist_m)
            predictions_df = pd.DataFrame({'timestamp': dates, 'ws': ws, 'ws-adjusted': ws})    
            return predictions_df
            
    model = loadMLmodel()
            
    #centroid x, centroid y, transformed XYp[x,y], rotated XYr[theta][x,y], L[theta], W[theta],XYti
    xc, yc, xyp, xyr, L, W, xyt = prepare_data(xy, xy_turbine, theta)
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

    outputs_0  = model.make_predictions(plot_test_data) 

    f=np.zeros(len(outputs_0))
    f=(outputs_0[:,0])*(wss[:])*np.power((plot_test_data[:,0])/z_turbine,0.143)#*(1.-eps[j])
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

    #print('LOM computation time :', total, ' sec')

    #predictions_df = pd.DataFrame({'timestamp': dates, 'linear':upl, 'nonlinear':upnl})
    #predictions_df['wtk'] = ws 
    
    predictions_df = pd.DataFrame({'timestamp': dates, 'ws': ws, 'ws-adjusted': upnl})    
    
    # end of vectorized version
    
    #return predictions_df
    # Clean up ANL's output
    #return predictions_df.rename(columns={"wtk": "ws", "nonlinear": "ws-adjusted"}).drop(columns=["linear"])
    
    return predictions_df