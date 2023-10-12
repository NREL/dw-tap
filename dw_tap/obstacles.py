import pandas as pd
import numpy as np
import os
import geopandas as gpd
from dw_tap.data_processing import filter_obstacles

class AllObstacles(object):

    def __init__(self, data_dir="", types=["bergey", "oneenergy"], debug=True):
        
        self.data_dir = data_dir
        
        if not data_dir:
            raise ValueError("data_dir needs to be set. Provide the path to the directory with DW-TAP data.")
        
        self.types = types
        
        # Create a dictionary of dictionaries: "bergey|onenergy": {<tid>: <dataframe with obstacles>}
        self.obstacles = {}
        
        # The same as above but limited to 100m around each turbine
        self.obstacles_100m = {}        
        
        for t in self.types:
                
            if t == "bergey":
                bergey_obstacles = {}
                bergey_obstacles_100m = {}
            
                index_path = os.path.join(self.data_dir, "01_bergey_turbine_data/bergey_sites.csv")
                index = pd.read_csv(index_path)

                # Get all site IDs
                selected = index["APRS ID"].tolist()

                # Remove 2 sites that currently don't have obstacle descriptions with the heights based on lidar data
                selected = [x for x in selected if not(x in ["t007", "t074"])]

                for tid in selected:

                    index_row = index[index["APRS ID"] == tid].iloc[0]
                    z_turbine = index_row["Hub Height (m)"]
                    lat = index_row["Latitude"]
                    lon = index_row["Longitude"]

                    obstacle_data_dir = os.path.join(self.data_dir, "01_bergey_turbine_data/3dbuildings_geojson")
                    obstacle_data_file = "%s/%sv3.json" % (obstacle_data_dir, tid)

                    
                    if os.path.exists(obstacle_data_file): 
                        raw_data = gpd.read_file(obstacle_data_file)
                        if debug:
                            # Show warnings if obs heght > z_turbine
                            obstacle_df = filter_obstacles(tid, 
                                                           raw_data, 
                                                           include_trees=True, 
                                                           turbine_height_for_checking=z_turbine,
                                                           version=3)
                            
                            # Same as above but limited to 100m
                            obstacle_df_100m = filter_obstacles(tid, 
                                                           raw_data, 
                                                           include_trees=True, 
                                                           turbine_height_for_checking=z_turbine,
                                                           limit_to_radius_in_m=100.0,
                                                           turbine_lat_lon=(lat, lon),
                                                           version=3)
                            
                        else:
                            # Don't show height warnings
                            obstacle_df = filter_obstacles(tid, 
                                                   raw_data, 
                                                   include_trees=True,
                                                   version=3)
                            
                            # Same as above but limited to 100m
                            obstacle_df_100m = filter_obstacles(tid, 
                                                           raw_data, 
                                                           include_trees=True,
                                                           limit_to_radius_in_m=100.0,
                                                           turbine_lat_lon=(lat, lon),
                                                           version=3)
                            
                        obstacle_df["tid"] = tid
                        bergey_obstacles[tid] = obstacle_df
                        
                        obstacle_df_100m["tid"] = tid
                        bergey_obstacles_100m[tid] = obstacle_df_100m

                    else:
                        print("Can't access: %s. Skipping" % obstacle_data_file)
                
                if debug:
                    total_count = np.sum([len(v) for k,v in bergey_obstacles.items()])
                    print("Loaded info for %d obstacles for %d sites from: %s" % (total_count, len(selected), obstacle_data_dir))
                    print("Site: # of obstacles --", {k:len(v) for k,v in bergey_obstacles.items()})
                    
                self.obstacles[t] = bergey_obstacles
                self.obstacles_100m[t] = bergey_obstacles_100m
            
            elif t == "oneenergy":
                
                oneenergy_obstacles = {}
                oneenergy_obstacles_100m = {}
                
                index_path = os.path.join(self.data_dir, "01_one_energy_turbine_data/OneEnergyTurbineData.csv")
                index = pd.read_csv(index_path)

                # Select sites with lidar data
                selected = ['p1w1', 'p1w2', 'p1z1', 'p1z2', 'p1z3', 'p1z4', 'p1z5', 'p1z6', 'p1v1', 'p1v2', 'p3wtg1', 'p4w1', 'p4w2', 'p4w3', 'p5w1', 'p6l1', 'p6l2', 'p6l3']

                for tid in selected:

                    index_row = index[index["APRS ID"] == tid].iloc[0]
                    z_turbine = index_row["Hub Height (m)"]
                    lat = index_row["Latitude"]
                    lon = index_row["Longitude"]
                    
                    # E.g., 'p1w1' -> 'p1'
                    site_id = tid[:2]

                    obstacle_data_dir = os.path.join(self.data_dir, "01_one_energy_turbine_data/3dbuildings_geojson")
                    
                    #obstacle_data_file = "%s/%sv2.json" % (obstacle_data_dir, site_id)
                    # Switched to v3 to use the latest
                    obstacle_data_file = "%s/%sv3.json" % (obstacle_data_dir, site_id)

                    
                    if os.path.exists(obstacle_data_file):

                        raw_data = gpd.read_file(obstacle_data_file)
                        if debug:
                            # Show warnings if obs heght > z_turbine
                            obstacle_df = filter_obstacles(site_id,
                                                           raw_data, 
                                                           include_trees=True, 
                                                           turbine_height_for_checking=z_turbine,
                                                           version=3)
                            
                            # Same as above but limited to 100m
                            obstacle_df_100m = filter_obstacles(site_id,
                                                           raw_data, 
                                                           include_trees=True, 
                                                           turbine_height_for_checking=z_turbine,
                                                           limit_to_radius_in_m=100.0,
                                                           turbine_lat_lon=(lat, lon),
                                                           version=3)
                        else:
                            # Don't show height warnings
                            obstacle_df = filter_obstacles(site_id,
                                                           raw_data, 
                                                           include_trees=True,
                                                           version=3)
                            
                            # Same as above but limited to 100m
                            obstacle_df_100m = filter_obstacles(site_id,
                                                           raw_data, 
                                                           include_trees=True, 
                                                           limit_to_radius_in_m=100.0,
                                                           turbine_lat_lon=(lat, lon),
                                                           version=3)
                            
                        obstacle_df["tid"] = tid
                        obstacle_df["site_id"] = site_id
                        oneenergy_obstacles[tid] = obstacle_df
                        
                        obstacle_df_100m["tid"] = tid
                        obstacle_df_100m["site_id"] = site_id
                        oneenergy_obstacles_100m[tid] = obstacle_df_100m

                    else:
                        print("Can't access: %s. Skipping" % obstacle_data_file)
                        
                if debug:
                    total_count = np.sum([len(v) for k,v in oneenergy_obstacles.items()])
                    print("Loaded info for %d obstacles for %d sites from: %s" % (total_count, len(selected), obstacle_data_dir))
                    print("Site: # of obstacles --", {k:len(v) for k,v in oneenergy_obstacles.items()})
                
                self.obstacles[t] = oneenergy_obstacles
                self.obstacles_100m[t] = oneenergy_obstacles_100m
            else:
                raise ValueError("Unsupported type included: %s. Currently supported: %s" % (t, str(["bergey", "oneenergy"])))
    
    
    def _get_with_checking(self, data, t, site):
        try:
            return data[t][site]
        except:
            raise ValueError("Combination of type and site (%s, %s) does not exist in the specified obstacle set." % (t, site))

    def get(self, t, site, obstacle_mode="all"):
        if obstacle_mode == "all" or obstacle_mode == "treesasbldgs":        
            return self._get_with_checking(self.obstacles, t, site)
            
        elif obstacle_mode == "treesasbldgs_100m":        
            return self._get_with_checking(self.obstacles_100m, t, site)
        
        elif obstacle_mode == "bldgsonly":        
            res = self._get_with_checking(self.obstacles, t, site)
            res = res[res["feature_type"] == "building"].reset_index(drop=True)
            return res
                             
        elif obstacle_mode == "bldgsonly_100m":  
            res = self._get_with_checking(self.obstacles_100m, t, site)
            res = res[res["feature_type"] == "building"].reset_index(drop=True)
            return res                
        else:
            raise ValueError("Obstacle mode %s is unsupported. Currently supported: ['all', 'treesasbldgs', 'treesasbldgs_100m', 'bldgsonly', 'bldgsonly_100m']" % obstacle_mode)                    

    def get_all_tall_obstacles(self, thresh_height=25.0):
        all_obs = []
        for t in self.types:
            all_obs.append(pd.concat([v for k,v in self.obstacles[t].items()]))
        all_obs = pd.concat(all_obs)
        
        return all_obs[all_obs["height"] > thresh_height]