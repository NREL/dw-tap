"""
dw_tap_lom_anl package includes: 
data_fetching.py with function getData(f, lat, lon, height, method=, power_estimate)
data_processing with functions geojson_toCoordinate(df_places, minx, maxx, miny, maxy, trees, porosity) and prepare_data(XY,XYt,theta)
loadMLmodel.py with function loadMLmodel()
power_output.py with function estimate_power_output(df, temp, pres) and class Bergey10(object) with methods windspeed_to_kw(cls, df) and reset_counters(cls)
process_geojson.py with functions get_obstacles(path) and get_candidate(path, candidate_id=0, footprint_size=1000) 

submodules include vector and interpolation
"""