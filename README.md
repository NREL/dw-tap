# DW TAP: TAP's Computational Pipeline

## Getting Started

Create conda environment (if it hasn't been created previously: `conda env create -f environment.yml`

If the environemnt has been created, activate it: `conda activate dw-tap`

Install the package (from inside the repo's main directory): `python setup.py install`. This step will need run only once for the created/activated environment (or every time the code inside the package is modified). Next time you activate the enironment, this package should be available.

## Running
For those with HSDS capabilities: 

run anl-lom-with-hsds-minimal

For those without HSDS capabilities (ANL): 

run anl-lom-no-hsds.ipynb

Two data options currently (11/13) exist. (1) CSV file "180_7years_1hourgranularity.csv" contains 7 years at 1 hour granularity for data at site 180. (2) CSV file "180_1year_12hourgranularity.csv" contains a single year at 12 hour time steps. Both can be read in within the notebooks at line: df = pd.read_csv(). Both datasets are within the data directory. 
## Contributors

