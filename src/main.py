import argparse
import os
import logging
from data_validator import validate_csv_file
from data_preprocessing import load_and_preprocess_data

def main():
    parser = argparse.ArgumentParser(description="AutoMLForge: Automated ML Pipeline")
    parser.add_argument("--data", "-d", required=True, help="Path to the CSV file.")
    parser.add_argument("--target", "-t", help="Name of the target column.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    parser.add_argument("--output", "-o", help="Path to the output directory.")
    args = parser.parse_args()

    try:
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output)

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                filename=os.path.join(args.output, 'automlforge.log') if args.output else None)
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                filename=os.path.join(args.output, 'automlforge.log') if args.output else None)

        logger = logging.getLogger(__name__)

        logger.info("Starting AutoMLForge pipeline.")

        validate_csv_file(args.data, logger)
        data = load_and_preprocess_data(args.data, args.target, logger)

        logger.info("Data loaded and preprocessed successfully.")
        print("Data loaded and preprocessed successfully:")
        print(data.head())

        # Further pipeline steps (model training, evaluation, etc.) will be added here

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
