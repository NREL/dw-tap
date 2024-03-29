# TAP's core functionality package

## Getting Started

Create conda environment (if it hasn't been created previously): `conda env create -f environment.yml`

If the environemnt has been created, activate it: `conda activate dw-tap`

Install the package (from inside the repo's main directory): `python setup.py install`. 

The last step will need to run only once for the created/activated environment (or every time the code inside the package is modified). Next time you activate the enironment, this package should be available.

## Testing

Activate the appropriate environment and execute the following command from this repo's root directory to run all tests (some may take a long time): 

```
export PYTHONWARNINGS="ignore" && pytest --log-cli-level=INFO tests/

```
To run a single test, run for example:
```
export PYTHONWARNINGS="ignore" && pytest --log-cli-level=INFO tests/pilowf_t024.py
```
