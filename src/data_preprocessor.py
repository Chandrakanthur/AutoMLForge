import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(data, target_column, logger):
    # (Same preprocess_data, handle_missing_data, handle_numerical_missing, and handle_outlier_iteration functions as before)
    logger.debug("Starting data preprocessing.")

    if target_column:
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in the DataFrame.")
        logger.debug(f"Target column: {target_column}")

    logger.debug("Handling missing data...")
    data = handle_missing_data(data, logger)

    logger.debug("Handling outliers...")
    data = handle_outliers(data, logger)

    logger.debug("Data preprocessing completed.")
    return data

def handle_missing_data(data, logger):
    # (Same missing data handling code as before)
    missing_percentages = data.isnull().sum() / len(data) * 100

    for col, percentage in missing_percentages.items():
        if percentage > 0:
            logger.debug(f"Missing percentage in '{col}': {percentage:.2f}%")
            if percentage <= 5:
                logger.debug(f"Removing rows with missing values in '{col}'.")
                data.dropna(subset=[col], inplace=True)
            else:
                if ptypes.is_numeric_dtype(data[col]):
                    logger.debug(f"Handling missing numerical values in '{col}'.")
                    data = handle_numerical_missing(data, col, logger)
                else:
                    logger.debug(f"Imputing missing categorical values in '{col}' with mode.")
                    data[col].fillna(data[col].mode()[0], inplace=True)
    return data

def handle_numerical_missing(data, col, logger):
    # (Same numerical missing data handling code as before)
    missing_percentage = data[col].isnull().sum() / len(data) * 100
    if missing_percentage > 0:
        if missing_percentage > 30: # If more than 30% missing, use median
            logger.debug(f"Imputing missing numerical values in '{col}' with median due to high missing percentage.")
            data[col].fillna(data[col].median(), inplace=True)
        else: # If less, try interpolation
            try:
                logger.debug(f"Attempting linear interpolation for missing values in '{col}'.")
                data[col].interpolate(method='linear', limit_direction='both', inplace=True)
                if data[col].isnull().sum() > 0: # If interpolation still has nulls, use median
                    logger.debug(f"Linear interpolation failed, using median for remaining nulls in '{col}'.")
                    data[col].fillna(data[col].median(), inplace=True)
            except Exception as e:
                logger.debug(f"Interpolation failed for '{col}': {e}. Using median imputation.")
                data[col].fillna(data[col].median(), inplace=True)

    return data

def handle_outliers(data, logger):
    """Handles outliers in numerical columns and generates a data insights graph."""
    outlier_data = pd.DataFrame()  # To store outlier data for the graph
    for col in data.select_dtypes(include=np.number).columns:
        logger.debug(f"Handling outliers in column: {col}")
        try:
            # Visualize outliers (Boxplot)
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=data[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

            # Identify outliers using IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

            if not outliers.empty:
                logger.debug(f"Found {len(outliers)} outliers in column: {col}")
                outlier_data[col] = outliers[col]  # Store outlier data
                data = handle_outlier_iteration(data, col, lower_bound, upper_bound, logger)
            else:
                logger.debug(f"No outliers found in column: {col}")

        except Exception as e:
            logger.error(f"Error handling outliers in column {col}: {e}")

    # Generate data insights graph if outliers are found
    if not outlier_data.empty:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=outlier_data)
        plt.title("Data Insights: Outlier Distribution")
        output_path = logging.getLogger().handlers[0].baseFilename  # Get the log file path
        if output_path:
            output_dir = os.path.dirname(output_path)
            plt.savefig(os.path.join(output_dir, "data_insights.png"))
            logger.debug(f"Data insights graph saved to: {os.path.join(output_dir, 'data_insights.png')}")
        else:
            logger.warning("Could not find output directory for data_insights.png")
        plt.show()

    return data

def handle_outlier_iteration(data, col, lower_bound, upper_bound, logger):
    # (Same handle_outlier_iteration function as before)
    original_data = data[col].copy()

    # 1. Capping (Winsorizing)
    capped_data = original_data.copy()
    capped_data[capped_data < lower_bound] = lower_bound
    capped_data[capped_data > upper_bound] = upper_bound
    data[col] = capped_data
    logger.debug(f"Capped outliers in column: {col}")

    # 2. Transformation (Log, Square Root)
    if (original_data > 0).all():  # Check for positive values
        transformed_data = np.log1p(original_data)
        outliers_transformed = transformed_data[(transformed_data < np.log1p(lower_bound)) | (transformed_data > np.log1p(upper_bound))]
        if outliers_transformed.empty:
            data[col] = transformed_data
            logger.debug(f"Transformed outliers in column: {col} using log transformation.")

    # 3. Replace with Median
    median_data = original_data.copy()
    median_data[(original_data < lower_bound) | (original_data > upper_bound)] = original_data.median()
    data[col] = median_data
    logger.debug(f"Replaced outliers with median in column: {col}")

    return data
