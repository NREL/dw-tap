### Summary

The imagined process is as follows:

1. acqiure and prepare data observation and wind toolkit data (Jenna)

   Note that Bergey data is production/power data and must be converted to windspeed

2. calculate bias correction parameters - linear fit (Caleb)

3. run models with and without Bias Correction (Dmitry)
 
  LOMS:
   - LANL (Dmitry)
   - ANL (Jenna)
   - ML? (Caleb)
   - Perrera (Caleb)
   - No LOM

  Scenarios:
   - WTK Only
   - WTK + Bias Correction
   - WTK with Obstacle Models
   - WTK + Bias Correction with Obstacle Models
  
4. subtract actuals from predictions, compute metrics make plots (Caleb)

  Metrics:
  
   - RMSE by site and 12x24
   - MAE by site and 12x24
   - Error quantiles/distribution by site and 12x24
 

### File formats and naming conventions

Step 1 will read in data from various subdirectories/sources and save prepared files in this directory as a set of CSVs with the following formats. First for observation files with the following naming convection and CSV with columns:

  observation_[bergey|oneenergy]_site_{x}_[h]m.csv
  
  datetime_yyyymmddhhmmss,windspeed_mps,winddir_deg
  
e.g.,

  observation_bergey_site_42_30m.csv
  
  datetime_yyyymmddhhmmss,windspeed_mps,winddir_deg
  20221001050505,10.3,260.1
  ...
  
Next, inflow (WIND toolkit) data:

  inflow_[wtk1|wtk2]_[bergey|oneenergy]_site_{x}_[h]m.csv
  
  datetime_yyyymmddhhmmss,windspeed_mps,winddir_deg

Step 2 will read the data from step 1, run the models and produce corresponding files as output, e.g.:

  anl_[bergey|oneenergy]_site_{x}_[h]m.csv
  
  datetime_yyyymmddhhmmss,windspeed_mps,winddir_deg
  
Step 3 will compute metrics and produce the summary files:

  results_summary.csv
  
  provider,site,height_m,model,metric,value
  
And, hour, month, sector summaries:

  results_hms_{metric}.csv
  
  provider,site,height_m,model,hour,month,sector_deg,value
  
