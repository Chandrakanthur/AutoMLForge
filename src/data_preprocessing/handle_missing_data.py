import pandas as pd
import pandas.api.types as ptypes

def handle_missing_data(data, logger):
    """Handles missing data in a pandas DataFrame with enhanced strategies."""
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
