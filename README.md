# Project Name: Kubeflow ML Pipeline for Shear Slowness Prediction

## Overview

This project is designed to create an extensive Kubeflow pipeline for machine learning (ML) model training and evaluation, specifically targeting the prediction of Shear Slowness measurements in wellbores. The pipeline incorporates multiple regression models, including Hist Gradient Boosting Regression, Linear Regression, Decision Tree Regression, Elastic Regression, Lasso Regression, and Polynomial Regression.

## Features

- **Modular and Parallel Processing:** The pipeline employs a modular and parallel processing approach for efficient resource utilization and scalability.
- **Model Variety:** A diverse set of regression models is utilized, offering a comprehensive comparison of their performance.
- **Data Preprocessing Scripts:** Included scripts (preprocess clean, preprocess entire, preprocess split) handle data cleaning, feature engineering, and data splitting.
- **Containerization with Docker:** Docker containers ensure reproducibility and simplify deployment across different environments.
- **Kubeflow Orchestration:** Kubeflow orchestrates different pipeline stages, providing a robust and scalable ML platform.

## Pipeline Structure

1. **Data Preprocessing:** Scripts for initial cleaning and transformation of input data.
2. **Regression Models:**
   - Decision Tree Regression
   - Elastic Net Regression
   - Hist Gradient Boosting Regression
   - Lasso Regression
   - Polynomial Regression
   - Random Forest Regression
   - Ridge Regression
3. **Show Results:** A component that aggregates and displays scores from each regression model, identifying the best-performing model.

## Usage

- **Setting Up the Environment:** Ensure Kubeflow and Docker are installed in your environment.
- **Pipeline Execution:** Run the provided YAML file (pipeline.yaml) using the Kubeflow Pipelines API to execute the entire ML workflow.

## Directory Structure

- `preprocess/`,`preprocess_clean/`, `preprocess_split` : Contains scripts for data preprocessing.
- `decision_tree_regression/`, `elastic_net_regression/`, etc.: Each directory contains scripts and configurations for a specific regression model.
- `show_results/`: Contains the script for displaying and comparing results of different regression models.

## Configuration

Adjust parameters in the YAML file to customize the pipeline according to your specific data and requirements.

## Dependencies

- Kubeflow
- Docker

## Results

After running the pipeline, scores of each regression model and the identified best model will be displayed. The pipeline also facilitates monitoring and tracking of the entire ML workflow.

## Contributors

- Mohamed Khairallah Gharbi
- Siva Sri Prasanna Maddila
- Michel Hatab
- Mouad EL MENSOUM
