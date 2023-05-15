import geopandas as gpd
import pandas as pd
import time
import multiprocessing as mp
import numpy as np
import os
import argparse
import glob
from dw_tap.lom import run_lom

# Handle argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--inputs_dir", action="store")
parser.add_argument("--index_file", action="store")
parser.add_argument("--output_filename_pattern", action="store")
parser.add_argument("--procs", action="store", type=int)
parser.add_argument("--output", action="store")

args = parser.parse_args()

inputs = [os.path.basename(f) for f in glob.glob("%s/*" % args.inputs_dir)]
#print(inputs)

# Assume filenames like: 't133-obstacles.json'
obstacle_tids = [f.split("-")[0] for f in inputs if "obstacles" in f]

# Assume filenames like: 't133-atmospheric.csv.bz2'
atmospheric_tids = [f.split("-")[0] for f in inputs if "atmospheric" in f]

index = pd.read_csv(args.index_file)

def process_site(tid):

    atmospheric_input = pd.read_csv(os.path.join(args.inputs_dir, "%s-atmospheric.csv.bz2" % tid))
    obs_df = gpd.read_file(os.path.join(args.inputs_dir, "%s-obstacles.json" % tid))
    
    row = index[index["APRS ID"] == tid].iloc[0]           
    lat = row["Latitude"]
    lon = row["Longitude"]
    z_turbine = row["Hub Height (m)"]
    xy_turbine = [np.array([lon, lat])]
    
    predictions_df = run_lom(atmospheric_input, \
                             obs_df, \
                             xy_turbine, z_turbine, \
                             check_distance=True)
    
    atmospheric_input["ws-adjusted"] = predictions_df["ws-adjusted"]
    
    dest_file = args.output_filename_pattern.replace("<TID>", tid)
    atmospheric_input.to_csv(dest_file, index=False)
    
    return atmospheric_input

def main():
    if set(obstacle_tids) != set(atmospheric_tids):
        print("Found mismatch between obstacle and atmospheric inputs (check TIDs).")
        return
    else:

        t_start = time.time()

        ctx = mp.get_context('spawn') # used for memory management: https://stackoverflow.com/questions/41240067/pandas-and-multiprocessing-memory-management-splitting-a-dataframe-into-multipl
        pool = ctx.Pool(args.procs)

        results = pool.map(process_site, atmospheric_tids)

        pool.close()
        pool.join()

        t_runtime = time.time() - t_start
        print('Runtime: %.2f (s)' % t_runtime)


if __name__ == "__main__":  # confirms that the code is under main function
    main()