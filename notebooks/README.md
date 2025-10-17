# Jupyter Notebooks

## Overview

This directory contains the Jupyter notebooks used for the exploratory data analysis, feature engineering, training, and evaluation of the machine learning models integrated into the system. Each notebook serves as a detailed record of the development process for a specific model.

## Directory Structure

The notebooks are organized into subdirectories, each corresponding to a specific machine learning model:

-   **`bitcoin/`**: Contains the notebook for the Bitcoin price prediction model.
-   **`flights/`**: Contains the notebook for the flight delay prediction model.
-   **`properties/`**: Contains the notebook for the property value prediction model.
-   **`movies_notebook.ipynb`**: The notebook used to develop the K-Nearest Neighbors (KNN) movie recommendation system.

Additionally, the directory includes:

-   **`constants.py`**: A utility script that defines shared constants, such as file paths for datasets, to ensure consistency across notebooks.

## Usage

To run these notebooks, please ensure you have installed all the required dependencies from the root `requirements.txt` file and have activated the Python virtual environment.

It is recommended to launch the Jupyter environment from the project's root directory to ensure that all relative paths, particularly those defined in [`notebooks/constants.py`](notebooks/constants.py), are resolved correctly.

```sh
# From the project root directory
source venv/bin/activate
jupyter lab
```