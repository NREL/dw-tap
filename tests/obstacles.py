import numpy as np
import pandas as pd
import h5pyd
import geopandas as gpd
from dw_tap.obstacles import AllObstacles
import warnings

import sys
sys.path.append("../scripts")
import dw_tap_data 

def test_read_in_all_obstacles():
    obs = AllObstacles(data_dir=dw_tap_data.path, types=["bergey", "oneenergy"], debug=False)

    total_len = len(obs.get_all_tall_obstacles(thresh_height=0.0))
    # Update the total number when dataset changes
    assert total_len == 7113, "Entire set of obstacles should include exactly 7113 obstacles"
  
def test_check_for_obstacles_taller_than_100m():  
    obs = AllObstacles(data_dir=dw_tap_data.path, types=["bergey", "oneenergy"], debug=False)

    tall_obstacles = obs.get_all_tall_obstacles(thresh_height=100.0)
    # Update this number after double-checking individual tall obstacles
    assert len(tall_obstacles) == 0, "The dataset should not include obstacles with height>100m; currently detected: %d such obstacles" % len(tall_obstacles)

def test_check_for_obstacles_taller_than_50m():
    obs = AllObstacles(data_dir=dw_tap_data.path, types=["bergey", "oneenergy"], debug=False)

    tall_obstacles = obs.get_all_tall_obstacles(thresh_height=50.0)
    # Update this number after double-checking individual tall obstacles
    assert len(tall_obstacles) == 0, "The dataset should not include obstacles with height>50m; currently detected: %d such obstacles" % len(tall_obstacles)

def test_check_different_obstacle_modes():
    tt = ["bergey", "oneenergy"] 
    obs = AllObstacles(data_dir=dw_tap_data.path, types=tt, debug=False)

    for t in tt:
        obs_subset = obs.obstacles[t]
        sites = obs_subset.keys()

        for s in sites:
            # Uncomment for debug messages; this can help ensure that all sites are actually getting checked
            # warnings.warn("Checking site: %s" % s)
  
            l1 = len(obs.get(t, s, "treesasbldgs"))
            l2 = len(obs.get(t, s, "treesasbldgs_100m"))
            l3 = len(obs.get(t, s, "bldgsonly"))
            l4 = len(obs.get(t, s, "bldgsonly_100m"))
        
            assert l2 <= l1, "checking site %s: |treesasbldgs_100m| should be <= |treesasbldgs|" % s
            assert l3 <= l1, "checking site %s: |bldgsonly| should be <= |treesasbldgs|" % s
            assert l4 <= l3, "checking site %s: |bldgsonly_100m| should be <= |bldgsonly|" % s
            assert l4 <= l2, "checking site %s: |bldgsonly_100m| should be <= |treesasbldgs_100m|" % s
