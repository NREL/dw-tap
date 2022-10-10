# -*- coding: utf-8 -*-
import numpy as np
import scipy
import geopandas as gpd
import matplotlib
import csv
import math
import pyproj
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from scipy import interpolate

# work with Geojson files:
def geojson_toCoordinate(df_places, minx, maxx, miny, maxy, trees, porosity):
    H = []
    XY = []
    eps = []
    i = 0
    for place in df_places.geometry: 
        if (( trees == True)):
            #if ((place.bounds[0]>minx) and (place.bounds[0]<maxx)): #100
                #if ((place.bounds[1]>miny) and (place.bounds[1]<maxy)): #100
            if np.isfinite(df_places.height[i]):
                    if (df_places.height[i]!='null'):
                        XY.append(np.array([_LatLon_To_XY(lat, lon) for lon, lat in zip(*place.exterior.coords.xy)]))
                        H.append(df_places.height[i])
                        if (( trees == True)):
                            # dmdu fix: No Type, Buildings only
                            eps.append(0.0)
        if (( trees == False)):
            
            #if df_places['type'][i] == 'building':
            # dmdu fix:
            if True:
                #if ((place.bounds[0]>minx) and (place.bounds[0]<maxx)): #100
                    #if ((place.bounds[1]>miny) and (place.bounds[1]<maxy)): #100
                if np.isfinite(df_places.height[i]):
                    if (df_places.height[i]!='null'):
                        XY.append(np.array([_LatLon_To_XY(lat, lon) for lon, lat in zip(*place.exterior.coords.xy)]))
                        H.append(df_places.height[i])
                        eps.append(0.0)
        i+=1
    return XY, H, eps

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
        xc, yc = _find_centroid(XY[i])
        #print('size place=', len(XY))
        xyc=_transform_tocent(XY[i], xc, yc)
        xyr=_rotate_coord(xyc, theta)
        L,W =_find_LW(xyr, np.shape(theta)[0])
        #print(L,W)
        xtr1=[]
        for j in range (len(XYt)):
            #print(turbine)
            xyti = _transform_tocentT(XYt[j], xc, yc)
            #print(xyti)
            #print(len(xyti))
            xtr=_rotate_coordT(xyti, theta)
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

def _find_centroid(XY):
    xc=(XY[:,0].max()+XY[:,0].min())/2.
    yc=(XY[:,1].max()+XY[:,1].min())/2.
    #print('xc: ', xc)
    return xc, yc

def _transform_tocent(XY, xc, yc):
    XYp=np.empty_like(XY)
    XYp[:,0]=XY[:,0]-xc
    XYp[:,1]=XY[:,1]-yc
    #print('XYp output: \n', XYp)
    return XYp

def _rotate_coord(XY, theta):
    XYt=np.empty([np.shape(theta)[0],np.shape(XY)[0], np.shape(XY)[1]])
    for i in range (np.size(theta)):
        XYt[i,:,0]=XY[:,0]*np.cos(np.deg2rad(theta[i]))-XY[:,1]*np.sin(np.deg2rad(theta[i]))
        XYt[i,:,1]=XY[:,0]*np.sin(np.deg2rad(theta[i]))+XY[:,1]*np.cos(np.deg2rad(theta[i]))
    return XYt.T

def _find_LW(XY, n):
    L = np.empty(n)
    W = np.empty(n)
    for i in range (n):
        W[i]=XY[0,:,i].max()-XY[0,:,i].min()
        L[i]=XY[1,:,i].max()-XY[1,:,i].min()
    return L, W

def _transform_tocentT(XY, xc, yc):
    XYp=np.empty_like(XY)
    if XY.ndim>1:
        XYp[:,0]=XY[:,0]-xc
        XYp[:,1]=XY[:,1]-yc
    else:
        XYp[0]=XY[0]-xc
        XYp[1]=XY[1]-yc
    return XYp

def _rotate_coordT(XY, theta):
    XYt=np.empty([np.shape(theta)[0],2])
    for i in range (np.size(theta)):
           XYt[i,0]=XY[0]*np.cos(np.deg2rad(theta[i]))-XY[1]*np.sin(np.deg2rad(theta[i]))
           XYt[i,1]=XY[0]*np.sin(np.deg2rad(theta[i]))+XY[1]*np.cos(np.deg2rad(theta[i]))
    return XYt


def _removeNegative(u):
    c= abs(u)+u
    return c/2


def _removeNans(u,s, b, x):
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

def _removeNans1(u,s, b, x,f,g):
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

def _removeNans2(u,s, b, x,f,g):
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

def _maskeddata(u, v, x):
    um = []
    vm = []
    i=0
    for xp in x:

        if xp < 0.:
            um.append(u[i])
            vm.append(v[i])
        i+=1
    return um, vm

def _maskeddata1(u, v, x):
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

def _LatLon_To_XY(Lat,Lon):
    """
    Input: Lat, Lon coordinates in degrees. 
    _LatLon_To_XY uses the albers projection to transform the lat lon coordinates in degrees to meters. 
    This is an internal function called by get_candidate.
    Output: Meter representation of coordinates. 
    """
    P = pyproj.Proj("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
    return P(Lon, Lat) #returned x, y note: lon has to come first here when calling