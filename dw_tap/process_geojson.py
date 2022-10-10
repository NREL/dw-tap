import shapely
import pyproj
import geopandas as gpd

def get_obstacles(path):
    """
    Reads map obstacles
    Input: path to geojson of obstacle polygons. 
    Output: geopandas reading of polygons. 
    """
    map_gdf_obstacles = gpd.read_file(path)
    return map_gdf_obstacles

def get_candidate(lat, lon, candidate_id=0, footprint_size=1000):
    """
    Input: path to geojson of candidate points. 
    footprint_size is the squared footprint of the turbine area.
    Output: lat lon coordinates in meters of the turbine location, 
    the maximum and minimum locations based on the footprint_size. 
    """
    
    x1_turbine = lon
    y1_turbine = lat
    
    x1_turbine, y1_turbine = _LatLon_To_XY(y1_turbine, x1_turbine)
    minx = x1_turbine - footprint_size 
    maxx = x1_turbine + footprint_size
    miny = y1_turbine - footprint_size
    maxy = y1_turbine + footprint_size
    
    return x1_turbine,y1_turbine,minx,maxx,miny,maxy


def _LatLon_To_XY(Lat,Lon):
    """
    Input: Lat, Lon coordinates in degrees. 
    _LatLon_To_XY uses the albers projection to transform the lat lon coordinates in degrees to meters. 
    This is an internal function called by get_candidate.
    Output: Meter representation of coordinates. 
    """
    P = pyproj.Proj("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")
    return P(Lon, Lat) #returned x, y note: lon has to come first here when calling