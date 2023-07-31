# Code and documentation
This repository contains the code used in my Master Thesis: *Seasonal Glacier Mass Balance Estimation Using AVHRR Optical Imagery*.

The data used in this project is partly publicly available:
- The SRTMGL30 v021 global 30 arcsec digital elevation model (DEM): [https://lpdaac.usgs.gov/products/srtmgl30v021/](https://lpdaac.usgs.gov/products/srtmgl30v021/)
- The outlines and attributes of glaciers from the Randolph Glacier inventory (version 6.0): [https://www.glims.org/RGI/](https://www.glims.org/RGI/)
- The *in situ* mass balance observations (glaciological method) of glaciers in the European Alps (RGI region "Central Europe"):[
  - WGMS: [https://wgms.ch/fogbrowser](https://wgms.ch/fogbrowser/)
  - GLAMOS (hydrological year product): [https://www.glamos.ch/en/downloads#mass-balance](https://www.glamos.ch/en/downloads#mass-balance)
- AVHRR LAC snow cover maps, 10-day composites of visible or on-ground snow cover

## Repository structure:
Here is the structure and the included files in each folder:

```bash
│   .gitignore
│   README.md
│
├───code
│       aggregate_netCDF.py
│       config.py
│       environment.yml
│       fig_3_scatterplots.py
│       fig_4_TimeseriesRegions.py
│       fig_5_CompareGeodetic.py
│       fig_6_ECDF_WOSM.py
│       fig_7_TemporalStability.py
│       functions_model.py
│       functions_plots.py
│       getAltSnowDistr.py
│       importer.py
│       main.py
│       main_calibrate.py
│       main_plot.py
│       main_validate.py
│       parallel_AltSnowDist.py
│       parallel_AltSnowDistr.sh
│
├───data
│   ├───avhrr
│   │   └───10_d_snowmasks
│   │
│   ├───geodata
│   │       CentralEurope_ge02km2_List.csv
│   │       dem_europe.tif
│   │       wkt.txt
│   │
│   └───insitu
│           all_insitu_annual.csv
│           all_insitu_summer.csv
│           all_insitu_winter.csv
│
├───intermediate
│   │   avhrr_scenes_per_season.csv
│   │
│   └───avhrr
│
└───output
    │
    └───figures
```

- The files `ignore.txt` can be ignored, as they were needed to update the folders in GitHub.
- Detailed information about how to run the code can be found below in the section #code.
- The folder `data` issplit into three subfolders containing:
  - `avhrr/10_d_snowmasks` contains AVHRR-derived snow cover maps as 10-day composites in daily netCDF (`.nc`) files with bands `scfv` and `scfg`.
    *Change the location of these files accordingly in the `config` script.*
  - `geodata`contains the auxiliary and supporting data for the snow map method, i.e. DEM, glacier attributes and centerpoints. 
    The file `wkt.txt` contains the command line code to resample the SRTM30 DEM to the AVHRR snow cover maps using `gdalwarp`.
  - `insitu` contains the seasonal mass balance observations copiled from WGMS and GLAMOS.
- The folder `intermedate` will host any intermediate files compiled and stored during processing. Contents of this folder should not be deleted and are not temporary files.
- The folder `output` will contain all figures, tables and other outputs produced in the scripts.

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)

## Code
To run the AVHRR snow map method, you will need:
a) a lot of time
b) to adjust the path to you r original AVHRR snow maps (`.nc` files)

Follow these steps to compute certain products:
1) Compute seasonal snow maps:
2) Compute altitudinal snow cover distributions:
3) Tune the model on in situ observaitons:
4) Generate mass balance time series of all 1228 glaciers analyzed in this study:
5) Generate figures:

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)

## Software requirements
**UPDATE VERSIONS**
The data pre-processing and data analysis was using Python 3.10.6, pandas (0.24.2), and GDAL 3.5.2. Newer versions of these software packages will likely work, but have not been tested. You can find the conda environments as `.yml` files in this repository. The file `drone.yml` can be used to build the environment neccessary for the drift correction. The file `environment.yml` in `code` can be used to build a conda environment for the scripts.

Code development and processing were carried out in Windows 10 (64 bit), but execution should (in theory) be platform independent.

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)

## Contact
Code development and maintenance: Nils Rietze ([nils.rietze@uzh.ch](nils.rietze@uzh.ch))

[to top](https://github.com/nrietze/avhrr_mass_balance/README.md)
