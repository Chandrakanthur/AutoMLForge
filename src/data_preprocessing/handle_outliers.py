import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

def handle_outliers(data, logger):
    # (Same handle_outliers and handle_outlier_iteration functions as before)
    outlier_data = pd.DataFrame()  # To store outlier data for the graph
    for col in data.select_dtypes(include=np.number).columns:
        logger.debug(f"Handling outliers in column: {col}")
        try:
            # Visualize outliers (Boxplot)
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=data[col])
            plt.title(f"Boxplot of {col}")
            plt.close() # Close plot to avoid showing
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
        plt.close()
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
