1. To get the necessary data for this directory, you must download per-turbine data into this directory using A2E/DAP:

https://a2e.energy.gov/ds/tap/turbine.aprs.00

You will likely be required to use a script to download the (very many, small) files.

2. Next, you can run 01_consolodate_files.sh which will move the many files into per-turbine directories and then combine the data into per-turbine files in this directory named like t042.txt. Though these have a .txt extension, they are in CSV format.
