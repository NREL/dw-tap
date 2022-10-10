from dw_tap.data_processing import geojson_toCoordinate
from dw_tap.data_processing import prepare_data
from dw_tap.loadMLmodel import loadMLmodel

import numpy as np
import time
import pandas as pd

def run_lom(dates, ws, theta, df_places, xy_turbine, z_turbine, minx, maxx, miny, maxy): 
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
    return predictions_df