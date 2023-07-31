# Code and documentation
This repository contains the code used in my Master Thesis: *Seasonal Glacier Mass Balance Estimation Using AVHRR Optical Imagery*.

The data used in this project is partly publicly available:
- The SRTM 30 arcsec digital elevation model (DEM): **INSERT LINK**
- The outlines and attributes of glaciers from the Randolph Glacier inventory (version 6.0): [https://www.glims.org/RGI/](https://www.glims.org/RGI/)
- The *in situ* mass balance observations (glaciological method) of glaciers in the European Alps (RGI region "Central Europe"): [https://wgms.ch/fogbrowser/](https://wgms.ch/fogbrowser/)
- AVHRR LAC snow cover maps, 10-day composites of visible or on-ground snow cover

## Repository structure:
There are three code folders that serve for the different processing steps in this study. Here is the structure and the included files in each folder:

**INSERT TREE**

```bash
├───code
│
└───data
```

- The scripts in `analysis` are used to generate the main and supplementary figures as well as the supporting tables.
- The scripts in `classification` are used to prepare the drone imagery and run the random forest classification.
- The scripts in `thermal_drift_correction` are used to remove the temperature drift for all images in a thermal flight.
- The folder `data` is empty and should contain the data that can be downloaded from Zenodo (see link on top).

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)

## Software requirements
**UPDATE VERSIONS**
The data pre-processing and data analysis was using Python 3.10.6, pandas (0.24.2), and GDAL 3.5.2. Newer versions of these software packages will likely work, but have not been tested. You can find the conda environments as `.yml` files in this repository. The file `drone.yml` can be used to build the environment neccessary for the drift correction. The file `environment.yml` in `code` can be used to build a conda environment for the scripts.

Code development and processing were carried out in Windows 10 (64 bit), but execution should (in theory) be platform independent.

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)

## Contact
Code development and maintenance: Nils Rietze ([nils.rietze@uzh.ch](nils.rietze@uzh.ch))

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)
