from dw_tap.data_processing import geojson_toCoordinate
from dw_tap.data_processing import prepare_data
#from dw_tap.loadMLmodel import loadMLmodel
from dw_tap.LOMvectorized import loadMLmodel
from dw_tap.data_processing import _LatLon_To_XY
#from tensorflow.python.ops.numpy_ops import np_config #DKF
import numpy as np
import time
import pandas as pd
import pyproj
#New bug DKF
#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()

from windrose import WindroseAxes

import matplotlib.pyplot as plt

#np_config.enable_numpy_behavior()



superposition = 3.0 #[1: linear, 2: nonlinear, 3: PILOWFv0.2]
steepness = 10.0 #PILOWF parameter: do not change it



def compute_superpositionCoefficients(ranked_wXsr, ranked_wHsr):
    n_lines, n_elements = ranked_wXsr.shape
    
    # Extend the dimensions for broadcasting
    rX = ranked_wXsr[:, :, None]
    rH = ranked_wHsr[:, :, None]
    rH_next = ranked_wHsr[:, None, :]
    
    # Calculate the terms in the summation
    diff = rX - rX.transpose((0, 2, 1))
    denominator = 5 * rH + 2 * rH_next
    terms = np.maximum(0, (1 - diff / denominator) * rH_next / rH)
    
    # Sum terms along the last dimension for dx calculation
    dx_sum = np.triu(terms).sum(axis=-1)
    dx = np.maximum(0, 2 - dx_sum)
    
    # Special calculation for the last element of dx
    if (n_elements)>2:
        last_elem_diff = 1 - (-ranked_wXsr[:, -2] + ranked_wXsr[:, -1]) / (5 * ranked_wHsr[:, -1] + 2 * ranked_wHsr[:, -1])
        #dx[:, -1] = np.minimum(1, last_elem_diff * ranked_wHsr[:, -1] / ranked_wHsr[:, -2])
        dx[:, -1] = np.maximum(0, np.minimum(1, last_elem_diff * ranked_wHsr[:, -1] / ranked_wHsr[:, -2]))
 
    print(dx)
    return dx    

def sort_based_on_distance(wXr, wHsr):
    sorted_indices = np.argsort(wXr, axis=0)  # Get the indices that would sort each line
    ranked_indices = np.flip(sorted_indices, axis=0)  # Reverse the order of the indices
    ranked_wXr = np.take_along_axis(wXr, ranked_indices, axis=0)  # Arrange elements based on ranked indices
    ranked_wHsr = np.take_along_axis(wHsr, ranked_indices, axis=0)  # Arrange elements of wHsr based on ranked indices
    return ranked_wXr, ranked_wHsr

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
    
    #plt.scatter(xc,yc)
    

    plot_test_data = np.zeros((len(L[0])*len(L)*len(xy_turbine),6))
    dimensioned_test_data = np.zeros((len(L[0])*len(L)*len(xy_turbine),6)) #used for PILOWF v0.2 superposition

    wss=np.zeros(len(L[0])*len(L)*len(xy_turbine))
    kk=0
    for i in range(len(xy_turbine)): #loop over turbines number#
        for j in range (len(L)):    #loop over building number#
            for k in range (len(L[0])): #loop over theta
                #PILOWF model inputs
                plot_test_data[kk,0] = H[j]/H[j]   #H changes between objects
                plot_test_data[kk,1] = W[j][k]/H[j]   #W alters with theta
                plot_test_data[kk,2] = L[j][k]/H[j]   #L alters with theta
                plot_test_data[kk,3] = (xyt[j][i][k,1]/H[j]) #s_turbine: alters with theta stramwise direction
                plot_test_data[kk,4] = (xyt[j][i][k,0])/H[j]   #w_turbine: alters with theta -spanwise direction
                
                #print(np.sqrt(xyt[j][i][k,1]*xyt[j][i][k,1]+xyt[j][i][k,0]*xyt[j][i][k,0])/H[j], H[j],i,j,k)
                plot_test_data[kk,5] = z_turbine/H[j]  #z[:]: constant     
                           
                #Only used for PILOWF v 0.2 superposition due to different nondimensionalisation
                dimensioned_test_data[kk,0] = H[j]   #H changes between objects
                dimensioned_test_data[kk,1] = W[j][k]   #W alters with theta
                dimensioned_test_data[kk,2] = L[j][k]   #L alters with theta
                dimensioned_test_data[kk,3] = (xyt[j][i][k,1]) #s_turbine: alters with theta stramwise direction
                dimensioned_test_data[kk,4] = (xyt[j][i][k,0])   #w_turbine: alters with theta -spanwise direction               
                #print(np.sqrt(xyt[j][i][k,1]*xyt[j][i][k,1]+xyt[j][i][k,0]*xyt[j][i][k,0])/H[j], H[j],i,j,k)
                dimensioned_test_data[kk,5] = z_turbine  #z[:]: constant
                           
                
                wss[kk]=ws[k]
                #plot_test_data[kk,6] = 0.0  #z[:]: constant

                kk=kk+1

                
                

    t1 = time.time()
    total = t1-t0

    outputs_0  = model.make_predictions(plot_test_data) 

    

    factors=(1. / (1. + np.exp(steepness*(plot_test_data[:,3] - 12.5))))*(1 / (1 + np.exp(steepness*(np.abs(plot_test_data[:,4]) - 4.0))))*(1 / (1 + np.exp(steepness*(np.abs(plot_test_data[:,5]) - 4.0))))
    
    f=np.zeros(len(outputs_0))
    f=(outputs_0[:,0]*factors[:])*(wss[:])*np.power((plot_test_data[:,0])/z_turbine,0.143)#*(1.-eps[j])
    out2=f.reshape(len(L),len(L[0])).T

    
    
    #fnl=np.zeros(ws[:])
    fl =out2.sum(axis=1)
    fnl1=out2[:]*out2[:]
    fnl =fnl1.sum(axis=1)
    
    if superposition == 1.0:
        upnl = ws[:]+fl[:]
        print('We used linear superposition')
    if superposition == 2.0:
        upnl = ws[:] + np.sqrt(fnl[:])
        print('We used non-linear superposition')
    if superposition  == 3.0:
        dx_buildings=dimensioned_test_data[:,3].reshape(len(L),len(L[0])).T
        H_buildings=dimensioned_test_data[:,0].reshape(len(L),len(L[0])).T
        ranked_wXsr, ranked_wHsr=sort_based_on_distance(dx_buildings, H_buildings)
        ranked_wXsr, ranked_wf=sort_based_on_distance(dx_buildings, out2)
        
        if len(L[0]) == 1:
            out3 = ranked_wf[:]
        else:
            dx=compute_superpositionCoefficients(ranked_wXsr, ranked_wHsr)
            out3=ranked_wf[:]*dx[:]
        fl3 =out3.sum(axis=1)
        upnl = ws[:] + fl3
        print("We used PILOWF v0.2 superposition")
        
#    upnl = ws[:]+fnl[:]
#    upl = ws[:]+fl[:]

    #fnlsum[i,k] =np.sqrt(np.sum(f[i,:,k]*f[i,:,k]))            

    t1 = time.time()
    total = t1-t0

    #print('LOM computation time :', total, ' sec')

    #predictions_df = pd.DataFrame({'timestamp': dates, 'linear':upl, 'nonlinear':upnl})
    #predictions_df['wtk'] = ws 
    predictions_df = pd.DataFrame({'timestamp': dates, 'ws': ws, 'ws-adjusted': upnl})    

    output_df = pd.DataFrame({'x': plot_test_data[:,3],  'y': plot_test_data[:,4], 'fprime': f})    

    
    # end of vectorized version
#    plt.figure
#    ax = WindroseAxes.from_ax()
#    ax.bar(theta, ws, normed=True, opening=0.8, edgecolor='white')
#    ax.set_legend()
#    plt.savefig("Windrose.png", dpi=900)
    #return predictions_df
    # Clean up ANL's output
    #return predictions_df.rename(columns={"wtk": "ws", "nonlinear": "ws-adjusted"}).drop(columns=["linear"])
    
    return predictions_df