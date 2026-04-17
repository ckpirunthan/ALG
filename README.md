# African Lovegrass Segmentation Using UAV Multispectral Imagery and Artificial Intelligence

## Project Overview

This repository contains the code developed for African Lovegrass (ALG) segmentation using UAV-based multispectral imagery and artificial intelligence. The project focuses on preparing labeled raster data, extracting spectral samples, training machine learning and deep learning models, and generating prediction maps for invasive weed detection.

The workflow supports data preparation from orthomosaic raster files, model training using multiple algorithms, and pixel-wise prediction over new multispectral UAV imagery. The repository mainly includes scripts for:

- converting polygon labels into point samples,
- extracting raster values from sampled points,
- loading seasonal datasets,
- training AI models, and
- generating spatial prediction maps.

This codebase supports the study on African Lovegrass segmentation using AI and UAV remote sensing and is intended for reproducible model development and testing.

## Repository Structure
This repository contains Python files organized into two main folders: `data` and `models`.

- **Data**: Contains labeled and extracted data from raster orthomosaic files.
- **Models**: Contains models developed from various algorithms.

#### File Descriptions

- **`convert_polygon_to_points.py`**: This script takes a polygon geopackage and converts it into points.
- **`sample_raster_from_points.py`**: This script reads the geopoints and takes samples from the raster file.
- **`seasonXloader.py`**: This file loads data from season X (site 1) stored within the `data` directory, performs data augmentation, and converts it into Pandas dataframes: `df_alg`, `df_nonalg`, and `df_nonveg`.
- **`seasonYloader.py`**: This file loads data from season Y (site 4) stored within the `data` directory, performs data augmentation, and converts it into Pandas dataframes: `df_alg`, `df_nonalg`, and `df_nonveg`.
- **`trainingCNN.py`**: Trains and stores Convolutional Neural Network (CNN) models using the provided dataframes.
 ![CNN model](https://github.com/ckpirunthan/ALG/blob/main/CNN%20model.jpg)
- **`trainingXGBoost.py`**: Trains and stores XGBoost models using the provided dataframes.
- **`trainingRF.py`**: Trains and stores Random Forest models using the provided dataframes.
- **`trainingSVM.py`**: Trains and stores Support Vector Machine (SVM) models using the provided dataframes.
- **`CNN_Prediction_map.py`**: Reads Multispectral UAV-generated orthomosaic images (raster file bands should be in the order blue, green, red, red-edge, and NIR) and predicts each pixel location using the developed CNN model.
- **`XGBoost_prediction_map.py`**: Reads Multispectral UAV-generated orthomosaic images and predicts each pixel location using the developed XGBoost model.
- - **`CNN_Prediction_map.py`**: Reads Multispectral UAV-generated orthomosaic images (raster file bands should be in the order blue, green, red, red-edge, and NIR) and predicts each pixel location using the developed CNN model.
- **`XGBoost_prediction_map.py`**: Reads Multispectral UAV-generated orthomosaic images (raster file bands should be in the order blue, green, red, red-edge, and NIR) and predicts each pixel location using the developed XGBoost model.
  ![Predicition using CNN](https://github.com/ckpirunthan/ALG/blob/main/Prediciton%20using%20CNN.png)
  
These scripts are essential for generating and preparing the data files used in this project.

## Requirements

The project is implemented in Python and uses common geospatial, machine learning, and deep learning libraries. Depending on the script, required packages may include:

- `numpy`
- `pandas`
- `geopandas`
- `rasterio`
- `scikit-learn`
- `xgboost`
- `tensorflow` / `keras`
- `matplotlib`

## Input Data Notes

For prediction scripts, the input raster should be a multispectral orthomosaic with the following band order:

1. Blue  
2. Green  
3. Red  
4. Red-edge  
5. NIR  

Incorrect band ordering may lead to unreliable predictions.

## Citation

If you use this repository in your research, please cite:

**APA style**  
Keerthinathan, P., Amarasingam, N., Kelly, J. E., Mandel, N., Dehaan, R. L., Zheng, L., Hamilton, G., & Gonzalez, F. (2024). *African Lovegrass Segmentation with Artificial Intelligence Using UAS-based Multispectral and Hyperspectral Imagery*. *Remote Sensing, 16*(13), 2363. https://doi.org/10.3390/rs16132363

**IEEE style**  
P. Keerthinathan, N. Amarasingam, J. E. Kelly, N. Mandel, R. L. Dehaan, L. Zheng, G. Hamilton, and F. Gonzalez, “African Lovegrass Segmentation with Artificial Intelligence Using UAS-based Multispectral and Hyperspectral Imagery,” *Remote Sensing*, vol. 16, no. 13, p. 2363, 2024, doi: 10.3390/rs16132363.

**BibTeX**
```bibtex
@article{keerthinathan2024african,
  title   = {African Lovegrass Segmentation with Artificial Intelligence Using UAS-based Multispectral and Hyperspectral Imagery},
  author  = {Keerthinathan, Pirunthan and Amarasingam, Narmilan and Kelly, Jane E. and Mandel, Nicolas and Dehaan, Remy L. and Zheng, Lihong and Hamilton, Grant and Gonzalez, Felipe},
  journal = {Remote Sensing},
  volume  = {16},
  number  = {13},
  pages   = {2363},
  year    = {2024},
  doi     = {10.3390/rs16132363}
}
