import os
import pandas as pd

def validate_csv_file(file_path, logger):
    """Validates the input CSV file."""
    logger.debug(f"Validating file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".csv"):
        raise ValueError("Invalid file format. Only CSV files are supported.")

    try:
        pd.read_csv(file_path)
    except pd.errors.ParserError as e:
        raise ValueError(f"Error reading CSV file: {e}")

    logger.debug(f"File validated successfully.")
