import argparse
import os
import logging
import pandas as pd
from data_validator import validate_csv_file
from data_preprocessing.data_pre_processing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas.api.types as ptypes
import yaml
from tpot import TPOTClassifier, TPOTRegressor
import stopit

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
        data = pd.read_csv(args.data)
        data = preprocess_data(data, args.target, logger)

        if args.target:
            target_column = args.target
        elif "target" in data.columns:
            target_column = "target"
        elif "target_column" in data.columns:
            target_column = "target_column"
        else:
            target_column = None

        if target_column:
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
            training_config = config["training"]

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=training_config["test_size"],
                                                                random_state=training_config["random_state"])

            # AutoML with TPOT and timeout
            if ptypes.is_numeric_dtype(y):
                tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=training_config["random_state"])
            else:
                tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=training_config["random_state"])

            try:
                with stopit.threading_timeoutable(default="timeout"):
                    tpot.fit(X_train, y_train, timeout=600)  # Timeout set to 600 seconds (10 minutes)
                logger.info(f"TPOT best pipeline: {tpot.fitted_pipeline_}")
            except stopit.TimeoutException:
                logger.warning("TPOT training timed out.")
                print("TPOT training timed out.")

            # Make Predictions
            y_pred = tpot.predict(X_test)

            # Evaluate Model
            if ptypes.is_numeric_dtype(y):
                mse = mean_squared_error(y_test, y_pred)
                logger.info(f"Mean Squared Error: {mse}")
                print(f"Mean Squared Error: {mse}")
            else:
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Accuracy: {accuracy}")
                print(f"Accuracy: {accuracy}")

        else:
            logger.warning("Target column not specified or found. Model training skipped.")
            print("Target column not specified or found. Model training skipped.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
