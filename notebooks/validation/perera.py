# perera.py
#
# Implementations of "classic" obstacle models from the literature.
#
# Note for those at NREL, there are examples of these functions in use 
# and some basic tests here: 
# https://github.nrel.gov/dw-tap/obstacle_ml_eaz/blob/master/03%20Traditional%20Empirical%20Models.ipynb
#
# Author: Caleb Phillips (caleb.phillips@nrel.gov)

import numpy as np
from math import exp, pow, log, tanh
import fiona
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, MultiPoint, shape
from shapely.ops import split, nearest_points

# Model from:
#
# M.D.A.E.S. Perera. Shelter Behind Two-Dimensional Solid and Porous Fences. 
# Journal of Wind Engineering and Industrial
# Aerodynamics, 9 (1981). 93-104.
#
# Implemented per description given in:
#
# Poul Astrup, Soren E Larsen. WAsP Engineering Flow Model for Wind over Land and Sea.
# Riso National Laboratory. Roskilde, Denmark. August 1999.
#
# Lars Landberg. Short-term Prediction of Local Wind Conditions. Roso-R-702. Riso
# National Laboratory, 1994. Ph.D. Thesis.

# x = distance downstream of obstacle (m)
# h = height of obstacle (m)
# z = height of target/turbine (m)
# po = porosity, eg 0 = solid, 0.2 = porous, 1.0 = no obstacle
# z0 = roughness height, height in meters of typical surface clutter, 
#      from https://en.wikipedia.org/wiki/Roughness_length, with roughness height = 10*roughness length
#      0.3 = open flat terrain & grass, 1.0 = low crops, 2.5 = high crops, 
#      5.0 = parks, 10.0 = suburb/forest, 20 = city center
@np.vectorize
def perera_factor(x,h,z,po=0.0,z0=0.3):
    kappa = 0.41 # see: https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_constant
    k = (2*pow(kappa,2))/(log(h/z0))
    vpe = 0.14 # velocity profile exponent
    n = z/h*pow((k*(x/h)),-1.0/(vpe+2.0))
    pe = 9.75*(1-po)*(h/x)*n*exp(-0.67*pow(n,1.5)) # this is the fraction of the reference wind speed (-delta u_z/u_h)
    return pe

# This version of the function refuses to make a prediction and returns None
# when perera would be unreasonable, i.e., when x/h < 5
@np.vectorize
def safe_perera_factor(x,h,z,po=0.0,z0=0.3):
    #print("safe_perera: {0},{1},{2}".format(x,h,z))
    if (h < 1) or (x/h < 5):
        return None
    kappa = 0.41 # see: https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_constant
    k = (2*pow(kappa,2))/(log(h/z0))
    vpe = 0.14 # velocity profile exponent
    n = z/h*pow((k*(x/h)),-1.0/(vpe+2.0))
    pe = 9.75*(1-po)*(h/x)*n*exp(-0.67*pow(n,1.5)) # this is the fraction of the reference wind speed (-delta u_z/u_h)
    return pe

# WaSP model
# b is the width of an obstacle centered and symmetric
# x is the distance to that obstacle
# a is a constant set "so that the wake behind a finite length obstacle spreads in
#   the same way as a free jet, i.e. the line through the points having half the velocity
#   deficit of that at the wake centerline for same x, this line shall have an angle
#   of 5 to 6 degree with the centerline"...so...um 0.5?
def wasp_geometric_factor_symmetric(b,x,a=0.5):
    return tanh((a*b)/(2.0*x))

# Same as the above but assume the object isn't centered, so it is ya long to the "right" 
#      and yb long to the "left"
def wasp_geometric_factor_asymmetric(ya,yb,x,a=0.5):
    return 0.5*tanh(a*(ya/x)) + 0.5*tanh(a*(yb/x))


def calculate_perera_features(point,buildings):
    ret = []
    for i in range(0,36):
        r = ((i*10.0 + 5.0)*2.0*np.pi)/360.0 # center of segment in radians
        y = np.sin(r)*1000.0                 # horiz. and vert. offset
        x = np.cos(r)*1000.0
        line = LineString([point, (point.x+x,point.y+y)])  # 1km line from turbine along segment center
        
        sret = {"pe": [],"ga": [], "gs": []}
        for b in buildings:
            s = shape(b["geometry"])
            if line.intersects(s):
                h = b["properties"]["height"]
                downwind_face = None
                dmin = None
                
                # make a minimum rotated rectangle and decompose into sides
                profile = s.minimum_rotated_rectangle.boundary
                points_to_split = MultiPoint([Point(x,y) for x,y in profile.coords[1:]])
                splitted = split(profile,points_to_split)
                
                for face in splitted:
                    # only consider faces (sides) that intersect with our ray
                    if line.intersects(face):
                        # keep the min distance side (face)
                        d = point.distance(face)
                        if (dmin is None) or (d < dmin):
                            dmin = d
                            downwind_face = face
                
                # for normal perrera this is good enough, we can calculate the P_e of this building
                # leaves p0 (porosity) at 0
                # leave roughness height at 0.3
                if (dmin is not None) and (h is not None):
                    sret["pe"].append(safe_perera_factor(dmin,h,15.0))
                
                    # now sort out the width of the face from the perspective of the 
                    # origin point. There's probably a more elegant way to do this, but what I'm
                    # more or less trying to do is:
                    # draw a line from the turbine to each end of the closest intersecting face
                    # the shorter of those lines is the side that's closer
                    # draw a line from that closer end to the other line, which should be a straight
                    #    line perpendicular to the azimuthal ray
                    # determine how much of that line is on either side of the azimuthal ray
                    #    in order to seed the geometric factor method
                    p1 = Point(downwind_face.coords[0])
                    p2 = Point(downwind_face.coords[1])
                    d1 = point.distance(p1)
                    d2 = point.distance(p2)
                    # p1 is closer, or it is perfectly perpendicular
                    if(d1 <= d2):
                        l = LineString([point,p2])
                        face_line = LineString(nearest_points(p1,l))
                        crossing = face_line.intersection(line)
                        y1 = crossing.distance(p1)
                        y2 = crossing.distance(l)
                    # p2 is closer
                    else:
                        l = LineString([point,p1])
                        face_line = LineString(nearest_points(p2,l))
                        crossing = face_line.intersection(line)
                        y1 = crossing.distance(p2)
                        y2 = crossing.distance(l)

                    sret["ga"].append(wasp_geometric_factor_asymmetric(y1,y2,dmin))  
                    sret["gs"].append(wasp_geometric_factor_symmetric(downwind_face.length,dmin))  
                    
        ret.append(sret)
    return ret

# infinite length obstacles
def perera(x,pdata):
    p = pdata[x[0]][int(x[1]/10.0) % 36]['pe']
    if (p is None) or (len(p) == 0):
        return x[2]
    p = np.where((p is None) or (p == np.array(None, dtype=object)),0,p)
    deficit = (np.array(p)*x[2]).sum()
    return x[2] - deficit

# centered, finite length obstacles
def perera2(x,pdata):
    try:
        p = pdata[x[0]][int(x[1]/10.0) % 36]['pe']
        g = pdata[x[0]][int(x[1]/10.0) % 36]['gs']
    except IndexError:
        return x[2]
    if (p is None) or (len(p) == 0):
        return x[2]
    p = np.where((p is None) or (p == np.array(None, dtype=object)),0,p)
    g = np.where((g is None) or (g == np.array(None, dtype=object)),0,g)
    deficit = (np.array(p)*np.array(g)*x[2]).sum()
    return x[2] - deficit

# asymmetric finite length obstacles
def perera3(x,pdata):
    p = pdata[x[0]][int(x[1]/10.0) % 36]['pe']
    g = pdata[x[0]][int(x[1]/10.0) % 36]['ga']
    if (p is None) or (len(p) == 0):
        return x[2]
    p = np.where((p is None) or (p == np.array(None, dtype=object)),0,p)
    g = np.where((g is None) or (g == np.array(None, dtype=object)),0,g)
    deficit = (np.array(p)*np.array(g)*x[2]).sum()
    return x[2] - deficit