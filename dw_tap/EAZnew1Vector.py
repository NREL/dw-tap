  # -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:34:19 2021

@author: demet
"""
import time
import numpy as np
import scipy
import geopandas as gpd
import matplotlib
import csv
import math
from windrose import WindroseAxes

import matplotlib.pyplot as plt
import pandas as pd

import EAZnew2

import sklearn.metrics
from scipy import interpolate
import time
import PILOWFlogo


def read_turbines(x1_turbine,y1_turbine,x2_turbine,y2_turbine,x3_turbine,y3_turbine,meters):
    turbinesarray =[]
    if ((meters==True)): 
        a=111000.
    else:
        a=1.
    turbinesarray.append(np.array([x1_turbine*a,y1_turbine*a]))
    if (x2_turbine !='nan'):
    #xy_turbine.append([x2_turbine*111000,y2_turbine*111000])
        turbinesarray.append(np.array([x2_turbine*a,y2_turbine*a]))
    if (x3_turbine !='nan'):
        turbinesarray.append(np.array([x3_turbine*a,y3_turbine*a]))
    return turbinesarray



# work with Geojson files:
def geojson_toCoordinate(df_places,minx,maxx,miny,maxy,meters, trees, porosity):
    H=[]
    XY=[]
    eps = []
    i=0
    for place in df_places.geometry:
        #print(df_places['type'][i], i)
        #if (df_places['type'][i] == 'building'):# or (df_places.types[i] == 'vegeration'):
        #if place.types[i] == 'building':
          #  print('I AM IN THE LOOP')
            #if (place.bounds[1]>.0088+53.31): #47 and 48
        if (( trees == True)):
            if ((place.bounds[0]>minx) and (place.bounds[0]<maxx)): #100
                if ((place.bounds[1]>miny) and (place.bounds[1]<maxy)): #100
                    if np.isfinite(df_places.height[i]):
                        if ((meters==True)): 
                            a=111000.
                        else:
                            a=1.
                        if (df_places.height[i]!='null'):
                            XY.append((np.array(place.exterior.coords.xy)*a).T)
                            H.append(df_places.height[i])
                            if (( trees == True)):
                                if df_places['type'][i] == 'building':
                                    eps.append(0.0)
                                if df_places['type'][i] == 'vegetation':
                                    eps.append(porosity)
        if (( trees == False)):
            if df_places['type'][i] == 'building':
                if ((place.bounds[0]>minx) and (place.bounds[0]<maxx)): #100
                    if ((place.bounds[1]>miny) and (place.bounds[1]<maxy)): #100
                        if np.isfinite(df_places.height[i]):
                            if ((meters==True)): 
                                a=111000.
                            else:
                                a=1.
                            if (df_places.height[i]!='null'):
                                XY.append((np.array(place.exterior.coords.xy)*a).T)
                                H.append(df_places.height[i])
                                eps.append(0.0)
        i+=1            
    return XY, H, eps


def find_centroid(XY):
 
    xc=(XY[:,0].max()+XY[:,0].min())/2.
    yc=(XY[:,1].max()+XY[:,1].min())/2.
    return xc, yc

def transform_tocent(XY, xc, yc):
    XYp=np.empty_like(XY)
    XYp[:,0]=XY[:,0]-xc
    XYp[:,1]=XY[:,1]-yc
    return XYp

def rotate_coord(XY, theta):
    XYt=np.empty([np.shape(theta)[0],np.shape(XY)[0], np.shape(XY)[1]])    
    for i in range (np.size(theta)): 
        XYt[i,:,0]=XY[:,0]*np.cos(np.deg2rad(theta[i]))-XY[:,1]*np.sin(np.deg2rad(theta[i]))
        XYt[i,:,1]=XY[:,0]*np.sin(np.deg2rad(theta[i]))+XY[:,1]*np.cos(np.deg2rad(theta[i]))
    return XYt.T

def find_LW(XY, n):
    L = np.empty(n)
    W = np.empty(n)
    for i in range (n):
        W[i]=XY[0,:,i].max()-XY[0,:,i].min()
        L[i]=XY[1,:,i].max()-XY[1,:,i].min()
    return L, W

def transform_tocentT(XY, xc, yc):
    XYp=np.empty_like(XY)
    if XY.ndim>1:
        XYp[:,0]=XY[:,0]-xc
        XYp[:,1]=XY[:,1]-yc
    else:
        XYp[0]=XY[0]-xc
        XYp[1]=XY[1]-yc
    return XYp

def rotate_coordT(XY, theta):
    XYt=np.empty([np.shape(theta)[0],2])   
    for i in range (np.size(theta)): 
           XYt[i,0]=XY[0]*np.cos(np.deg2rad(theta[i]))-XY[1]*np.sin(np.deg2rad(theta[i]))
           XYt[i,1]=XY[0]*np.sin(np.deg2rad(theta[i]))+XY[1]*np.cos(np.deg2rad(theta[i]))
    return XYt


def prepare_data(XY,XYt,theta):  
    XYp=[]
    XYr=[]
    xyc=[]
    xci=[]
    yci=[]
    Li=[] 
    Wi=[]
    XYti=[]
    for i in range(len(XY)):
        xc, yc = find_centroid(XY[i])
        #print('size place=', len(XY))
        xyc=transform_tocent(XY[i], xc, yc)
        xyr=rotate_coord(xyc, theta)
        L,W =find_LW(xyr, np.shape(theta)[0]) 
        #print(L,W)
        xtr1=[]
        for j in range (len(XYt)):
            #print(turbine)
            xyti = transform_tocentT(XYt[j], xc, yc)
            #print(xyti)
            #print(len(xyti))
            xtr=rotate_coordT(xyti, theta)
            xtr1.append(xtr)
        XYti.append(xtr1)            
        xci.append(xc)
        yci.append(yc)
        XYp.append(xyc)      
        XYr.append(xyr)
        Li.append(L)
        Wi.append(W)        
    return xci, yci, XYp, XYr, Li, Wi, XYti 
#centroid x, centroid y, transformed XYp[x,y], rotated XYr[theta][x,y], L[theta], W[theta],XYti

def removeNegative(u):
    c= abs(u)+u
    return c/2


def removeNans(u,s, b, x):
    v=[]
    w=[]
    p=[]
    d=[]

    i=0
    for up in u:
        if ((not math.isnan(u[i])) & (not math.isnan(b[i]))) :
            v.append(u[i])
            w.append(s[i])
            p.append(b[i])
            d.append(x[i])

        i+=1    
    return v, w, p, d




def removeNans1(u,s, b, x,f,g):
    v=[]
    w=[]
    p=[]
    d=[]
    k=[]
    l=[]
    i=0
    for up in u:
        if ((not math.isnan(u[i])) & (not math.isnan(b[i])) & (not math.isnan(f[i]))& (g[i]>0.)):
            v.append(u[i])
            w.append(s[i])
            p.append(b[i])
            d.append(x[i])
            k.append(f[i])
            l.append(g[i])
        i+=1    
    return v, w, p, d, k,l



def removeNans2(u,s, b, x,f,g):
    v=[]
    w=[]
    p=[]
    d=[]
    k=[]
    l=[]
    i=0
    for up in u:
        if ( (u[i]>2.0) & (u[i]<12.0)):
            v.append(u[i])
            w.append(s[i])
            p.append(b[i])
            d.append(x[i])
            k.append(f[i])
            l.append(g[i])
        i+=1    
    return v, w, p, d, k,l






def maskeddata(u, v, x):
    um = []
    vm = []
    i=0
    for xp in x:
        
        if xp < 0.:        
            um.append(u[i])
            vm.append(v[i])
        i+=1
    return um, vm
    
def maskeddata1(u, v, x):
    um = []
    vm = []
    i=0
    for xp in x:
        if xp > 0.:
            #if (((u[i] -u[i]) == 0.0) &&  ((v[i] -v[i]) == 0.0)):
             if ((not math.isnan(u[i]) ) &  (not math.isnan(v[i]))):  
                um.append(u[i])
                vm.append(v[i])
        i+=1
    return um, vm 





lms = ['.lm']#,'.old']#,'.lm','.lm2',]
models = ['LOM']#,'ML',]




turbines=[47, 48, 56, 79, 88, 94, 100, 107, 108, 125, 126, 161, 162, 163, 178, 182, 183, 227, 228, 250, 'MetTower']
#turbines=[79, 88, 94, 100, 107, 108, 125, 126, 161, 162, 163, 178, 182, 183, 227, 228, 250, 'MetTower']



# Trees treatment & Porosity
trees = False #True #False #True --> use trees #False --> don't use trees
porosity = 0.0




PILOWFlogo.print_logo()


for model in models:
    if model == 'LOM':
        import LOMvectorized
    #LOM warming up
        data = LOMvectorized.load_data('1x1x1-2-3-1x2x1i-cheb.dat') #data used for training
        lom = LOMvectorized.regression_model(data)
#lom.train_model() # train the model
        lom.test_model() # check performance on testing alone   
    if model == 'ML':
        import LOMML
        #LOMML warming up
        data = LOMML.load_data('1x1x1-2-3-1x2x1i-cheb.dat') #data used for training
        LOMML = LOMML.regression_model(data)
        #LOMML.train_model() # train the model
        LOMML.test_model() # check performance on testing alone
    for lm in lms:
        for turb in turbines:
            
            
            
            start1 = time.time()
            
            print("Computing LOM "+model+" for turbine:" + str(turb)+" and input"+ lm)
            df_places, ws, theta, x1_turbine,y1_turbine,minx,maxx,miny,maxy, dates, namesT=EAZnew2.dataRead(turb,lm)
            
            x2_turbine,y2_turbine,x3_turbine,y3_turbine='nan','nan','nan','nan'
            z_turbine = 15 # turbine height in [m]  
                #read the geometry. "True" changes units to [m], this is required because heights are in m
            xy,H, eps=geojson_toCoordinate(df_places,minx,maxx,miny,maxy,True, trees, porosity)
            xy1_turbine=read_turbines(x1_turbine,y1_turbine,x2_turbine,y2_turbine,x3_turbine,y3_turbine,False)
            xy_turbine=read_turbines(x1_turbine,y1_turbine,x2_turbine,y2_turbine,x3_turbine,y3_turbine,True)
            eps=np.array(eps) #make an array of porosities

            #centroid x, centroid y, transformed XYp[x,y], rotated XYr[theta][x,y], L[theta], W[theta],XYti
            xc,yc,xyp,xyr,L,W, xyt =prepare_data(xy,xy_turbine, theta)
            

            
            
            
            end1 = time.time()
            print("Time for preprocessing", end1 - start1," sec")



#A             H, W, L xyt[0,1],xyt[0,0],zturbine, eps


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
         


            if model == "LOM":
                outputs_0  = lom.make_predictions(plot_test_data) 
            if model == "ML":
                outputs_0  = LOMML.make_predictions(plot_test_data)            
                #print (plot_test_data[0,0],plot_test_data[0,1],plot_test_data[0,2], plot_test_data[0,3], plot_test_data[0,4], plot_test_data[0,5],outputs_0)
           #     f[i,j,:] = outputs_0*ws[:]*np.power(H[j]/15.,0.143)*(1.-eps[j])
                            #print(i,'/',len(xy_turbine)-1,',', k,'/',len(L[0])-1,',',j,'/',len(L)-1)
           #         fsum[i,k] =np.sum(f[i,:,k])
           #         fnlsum[i,k] =np.sqrt(np.sum(f[i,:,k]*f[i,:,k]))
           #         upnl[i,k]=ws[k]-fnlsum[i,k]
           #         upl[i,k]=ws[k]-fsum[i,k]
 
            f=np.zeros(len(outputs_0))
            f=(outputs_0[:,0])*(wss)*np.power((plot_test_data[:,0])/15.,0.143)#*(1.-eps[j])
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
            
            #save a CSV file for each turbine. Name: [geojsonname].turbine.[# of turbine (usually 0 or 1)].csv 
            #each CSV file has 4 columns:
            #count, date, u for linear superposition of the solutions, u for non linear superposition of the solutions    
            count=0
            with open(namesT+model+'_'+lm.replace(".","")+'_'+(str(dates[0]).replace(" ", "-")).replace(":", ";")+'_'+(str(dates[-1]).replace(" ", "-")).replace(":", ";")+'.csv', 'w', newline='') as csvfile:
                fieldnames = ['count','date', 'linear', 'nonlinear']
                writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(ws)):
                    count+=1
                    writer.writerow({'count':count,'date':dates[i],'linear':upl[i],'nonlinear':upnl[i]})
                    

  
    
    
