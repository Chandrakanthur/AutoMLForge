def generate_predict_script(output_dir, preprocess_called):
    script_content = """
import argparse
import os
import logging
import pandas as pd
import joblib
"""

    if preprocess_called:
        script_content += """
from data_preprocessing.data_pre_processing import preprocess_data
"""

    script_content += """
def predict(data_file, model_file, output_file, verbose, target_column=None):
    try:
        if verbose:
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')

        logger = logging.getLogger(__name__)

        logger.info("Starting prediction process.")

        # Load the trained full pipeline
        full_pipeline = joblib.load(model_file)
        logger.info(f"Full pipeline loaded from: {model_file}")

        # Load the new data
        new_data = pd.read_csv(data_file)
"""

    if preprocess_called:
        script_content += """
        # Preprocess the new data
        preprocessed_data = preprocess_data(new_data, target_column, logger)
        if target_column and target_column in preprocessed_data.columns:
            preprocessed_data = preprocessed_data.drop(target_column, axis=1)

        # Make predictions using the full pipeline
        predictions = full_pipeline.predict(preprocessed_data)
"""
    else:
        script_content += """
        # Make predictions using the full pipeline
        predictions = full_pipeline.predict(new_data)
"""

    script_content += """
        # Save predictions to a CSV file
        pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained full pipeline.")
    parser.add_argument("--data", "-d", required=True, help="Path to the new data CSV file.")
    parser.add_argument("--model", "-m", required=True, help="Path to the trained full pipeline file.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output CSV file for predictions.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    parser.add_argument("--target", "-t", help="Name of the target column.")
    args = parser.parse_args()

    predict(args.data, args.model, args.output, args.verbose, args.target)
"""

    output_path = os.path.join(output_dir, "predict.py") if output_dir else "predict.py"
    with open(output_path, "w") as f:
        f.write(script_content)
