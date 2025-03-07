# AutoMLForge
AutoMLForge: An automated machine learning pipeline for supervised learning tasks. This platform automates data preprocessing, model selection, hyperparameter tuning, and evaluation, streamlining the development of production-ready machine learning models.

# AutoMLForge: Automated Machine Learning Pipeline

AutoMLForge is a Python-based automated machine learning (AutoML) pipeline designed to streamline the process of building and deploying machine learning models. It automates key steps, including data validation, preprocessing, feature engineering, model selection, training, and prediction script generation.

## Features

* **Command-Line Interface (CLI):** Easy-to-use CLI for running the pipeline.
* **Data Validation:** Ensures data quality through validation checks.
* **Modular Data Preprocessing:** Comprehensive data preprocessing pipeline with various feature engineering techniques.
* **Automated Model Selection:** Utilizes TPOT for automated model selection and hyperparameter tuning.
* **Configuration Management:** Uses a `config.yaml` file for easy configuration of pipeline parameters.
* **Model Persistence:** Saves trained models using `joblib`.
* **Dynamic Prediction Script Generation:** Generates a `predict.py` script based on user satisfaction and preprocessing steps.
* **Full Pipeline Saving:** Ability to save the entire pipeline (preprocessing + model) for easy deployment.
* **Timeout Functionality:** Prevents long running trainings.
* **Regression and Classification Support:** Handles both regression and classification tasks.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/yourusername/AutoMLForge.git
    cd AutoMLForge
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    (Create a `requirements.txt` file with all dependencies `pip freeze > requirements.txt`)

## Usage

1.  Prepare your data in a CSV file.
2.  Configure the pipeline parameters in `config/config.yaml`.
3.  Run the pipeline using the following command:

    ```bash
    python main.py --data your_data.csv --target target_column --output output_dir [--verbose] [--download]
    ```

    * `--data`: Path to the CSV file.
    * `--target`: Name of the target column.
    * `--output`: Path to the output directory.
    * `--verbose`: Enable verbose output.
    * `--download`: Download the trained model and pipeline.

4.  After the model is trained, you'll be prompted to indicate your satisfaction. If you enter "yes," a `predict.py` script will be generated in the output directory.

5.  To make predictions on new data using the generated script:

    ```bash
    python predict.py --data new_data.csv --model full_pipeline.joblib --output predictions.csv [--verbose] [--target target_column]
    ```

## Configuration (config.yaml)

```yaml
feature_engineering:
  correlation_threshold: 0.8
  variance_threshold: 0.01
  polynomial_degree: 2
  log_transform_skew_threshold: 0.75
  binning:
    enabled: true
    num_bins: 10
    strategy: "quantile" # "uniform", "quantile", "kmeans"
  encoding:
    categorical_threshold: 10 # Number of unique values to consider a categorical feature
    default_encoding: "onehot" # "onehot", "label", "target"
  interaction_features:
    enabled: true
    pairs:
      - ["feature1", "feature2"] # Example, replace with actual feature names
      - ["feature3", "feature4"]
  feature_selection:
    enabled: true
    model_based:
      enabled: true
      num_features: 20

training:
  test_size: 0.2
  random_state: 42
  model:
    classification:
      name: "RandomForestClassifier"
      params:
        n_estimators: 100
    regression:
      name: "RandomForestRegressor"
      params:
        n_estimators: 100

Dependencies
Python 3.x
pandas
scikit-learn
TPOT
PyYAML
joblib
stopit
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

License
MIT License  https://github.com/Chandrakanthur/AutoMLForge/commit/94c0e652ee2fd2f390bfa15788176684d72c3424file

Author
M Chandrakanth Urs
Contact
chandrakanthurs123@gmail.com

Future Improvements
Hyperparameter tuning.
More model options.
Advanced evaluation metrics.
Automated model selection.
Deployment options.
GUI or web interface.
More logging and error handling.
Refactor the CustomPreprocessor.
