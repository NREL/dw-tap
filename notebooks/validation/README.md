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
   - Error quantiles/distribution by site
   - Average annual kwh error/difference by site and distribution of years
 
 5. analysis of error vs covariates - terrain, etc. (Lindsay)
 

### File formats and naming conventions

Step 1 will read in the data from various sources and format it such that it is ready to be read into the models.
All of the prepared data can be found in the folder: 02InputForModels
For the Bergey Data: 
   1) Navigate further to the Bergey folder
   2) The Bergey Folder contains the file "BergeyIndexSites.xslx" which has site index numbers (for file reference), turbine location, latitude, longitude, and turbine height, as well as the turbine type and period of generation with availabel output data. 
   3) The folder 02BuildingInputs contains geojson files outlining the buildings surrounding the turbine. Note: The building height for all buildings has been set to 5 meters and the building footprint has been manually drawn as of 3/9/23 by Jenna Ruzekowicz (jruzekow@nrel.gov). These geojson files will be named with "B" + [File Index].geojson where the [File Index] is the numerical value that applies to the Bergey turbine site and "B" indicates it is a Bergey site. File Index can be found in the "BergeyIndexSites.xslx" file. 
   4) The folder 02WTKInputs will contain the WTK extracted wind speed, wind direction, temperature and pressure data for the turbine location and height. These files will be in csv format with the following column names: "timestamp" "ws", "wd", "temp", "pressure". These files will be named "B" + [File Index] +"W".csv where the [File Index] is the numerical value that applies to the Bergey turbine site and "B" indicates it is a Bergey site and "W" indicates it is WTK data. File Index can be found in the "BergeyIndexSites.xslx" file.
   5) The folder 02MeasuredInputs will contain the measured wind speed, and wind direction. These files will be in csv format with the following column names: "ws", "wd". These files will be named "B" + [File Index] +"M".csv where the [File Index] is the numerical value that applies to the Bergey turbine site and "B" indicates it is a Bergey site and "M" indicates it is measured data. File Index can be found in the "BergeyIndexSites.xslx" file.
   
To run a model on a bergey site... 
