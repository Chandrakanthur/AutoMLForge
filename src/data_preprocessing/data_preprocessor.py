import pandas as pd
import logging
from .handle_missing import handle_missing_data
from .handle_outliers import handle_outliers
from .handle_skew import reduce_skewness
from .feature_engineering import feature_engineering_and_correlation
from .data_encoding import encode_categorical_data
from sklearn.preprocessing import StandardScaler
import yaml
import numpy as np

def preprocess_data(data, target_column, logger):
    """Preprocesses the input DataFrame."""
    logger.debug("Starting data preprocessing.")

    if target_column:
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in the DataFrame.")
        logger.debug(f"Target column: {target_column}")

    logger.debug("Handling missing data...")
    data = handle_missing_data(data, logger)

    logger.debug("Handling outliers...")
    data = handle_outliers(data, logger)

    # Load config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    fe_config = config["feature_engineering"]

    # Only execute if needed.
    if fe_config["polynomial_degree"] > 1 or fe_config["interaction_features"]["enabled"] or fe_config["feature_selection"]["enabled"] or data.select_dtypes(include=['object']).columns.any():
        logger.debug("Reducing skewness...")
        for col in data.select_dtypes(include=np.number).columns:
            if col != target_column:
                logger.debug(f"Reducing skewness in column: {col}")
                data = reduce_skewness(data, col, logger)

        logger.debug("Encoding categorical data...")
        data = encode_categorical_data(data, logger)

        logger.debug("Performing feature engineering and correlation analysis...")
        data = feature_engineering_and_correlation(data, logger)

    logger.debug("Scaling numerical data...")
    data = scale_numerical_data(data, logger)

    logger.debug("Data preprocessing completed.")
    return data

def scale_numerical_data(data, logger):
    """Scales numerical data using StandardScaler."""
    numerical_cols = data.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    logger.debug("Numerical data scaled using StandardScaler.")
    return data
