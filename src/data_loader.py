import pandas as pd
import os
import argparse
import logging

def load_csv_data(file_path, target_column=None, verbose=False, output_path=None):
    """
    Loads data from a CSV file, validates the file format, and handles target column.

    Args:
        file_path (str): The path to the CSV file.
        target_column (str, optional): The name of the target column. Defaults to None.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        output_path (str, optional): The path to the output directory. Defaults to None.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a CSV file or if there are issues reading the file.
        KeyError: If the target column is not found.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(output_path, 'data_loader.log') if output_path else None)
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=os.path.join(output_path, 'data_loader.log') if output_path else None)

    logger = logging.getLogger(__name__)

    try:
        logger.debug(f"Loading data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.lower().endswith(".csv"):
            raise ValueError("Invalid file format. Only CSV files are supported.")

        data = pd.read_csv(file_path)

        if target_column:
            if target_column not in data.columns:
                raise KeyError(f"Target column '{target_column}' not found in the CSV file.")
            logger.debug(f"Target column: {target_column}")

        return data

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error reading CSV file: {e}")
        raise ValueError(f"Error reading CSV file: {e}")
    except KeyError as e:
        logger.error(f"Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data from a CSV file.")
    parser.add_argument("--data", "-d", required=True, help="Path to the CSV file.")
    parser.add_argument("--target", "-t", help="Name of the target column.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    parser.add_argument("--output", "-o", help="Path to the output directory.")
    args = parser.parse_args()

    try:
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output)

        data = load_csv_data(args.data, args.target, args.verbose, args.output)
        print("Data loaded successfully:")
        print(data.head())
    except Exception:
        print("Data loading failed.")
