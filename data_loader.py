import pandas as pd
import os

def load_csv_data(file_path):
    """
    Loads data from a CSV file and validates the file format.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a CSV file or if there are issues reading the file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.lower().endswith(".csv"):
            raise ValueError("Invalid file format. Only CSV files are supported.")

        data = pd.read_csv(file_path)
        return data

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        raise ValueError(f"Error reading CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

# Example usage (for testing):
if __name__ == "__main__":
    try:
        # Replace 'sample_data.csv' with the actual path to your CSV file
        data = load_csv_data("sample_data.csv")
        print("Data loaded successfully:")
        print(data.head())  # Print the first few rows of the DataFrame
    except Exception:
        print("Data loading failed.")
