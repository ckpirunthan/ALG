### Project Overview

This repository contains Python files organized into two main folders: `data` and `models`.

- **Data**: Contains labeled and extracted data from raster orthomosaic files.
- **Models**: Contains models developed from various algorithms.

#### File Descriptions

- **`convert_polygon_to_points.py`**: This script takes a polygon geopackage and converts it into points.
- **`sample_raster_from_points.py`**: This script reads the geopoints and takes samples from the raster file.
- **`seasonXloader.py`**: This file loads data from season X (site 1) stored within the `data` directory, performs data augmentation, and converts it into a Pandas dataframe.
- **`seasonYloader.py`**: This file loads data from season Y (site 4) stored within the `data` directory, performs data augmentation, and converts it into a Pandas dataframe.

These scripts are essential for generating and preparing the data files used in this project.
