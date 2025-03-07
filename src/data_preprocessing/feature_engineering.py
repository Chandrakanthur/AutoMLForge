import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas.api.types as ptypes

def feature_engineering_and_correlation(data, logger):
    """Performs feature engineering and correlation analysis using config parameters."""
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    fe_config = config["feature_engineering"]

    logger.debug("Calculating correlation matrix.")
    corr_matrix = data.corr().abs()

    # Generate and save correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    output_path = logging.getLogger().handlers[0].baseFilename
    if output_path:
        output_dir = os.path.dirname(output_path)
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        logger.debug(f"Correlation matrix heatmap saved to: {os.path.join(output_dir, 'correlation_matrix.png')}")
    else:
        logger.warning("Could not find output directory for correlation_matrix.png")
    plt.close() # Close the plot to prevent showing it

    # Identify highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > fe_config["correlation_threshold"])]

    logger.debug(f"Highly correlated features to drop: {to_drop}")
    data.drop(to_drop, axis=1, inplace=True)

    # Polynomial Features
    if fe_config["polynomial_degree"] > 1:
        poly = PolynomialFeatures(degree=fe_config["polynomial_degree"])
        poly_features = poly.fit_transform(data.select_dtypes(include=np.number))
        poly_feature_names = poly.get_feature_names_out(data.select_dtypes(include=np.number).columns)
        data = pd.concat([data.drop(data.select_dtypes(include=np.number).columns, axis=1), pd.DataFrame(poly_features, columns=poly_feature_names)], axis=1)
        logger.debug(f"Polynomial features of degree {fe_config['polynomial_degree']} created.")

    # Interaction Features
    if fe_config["interaction_features"]["enabled"]:
        for pair in fe_config["interaction_features"]["pairs"]:
            if all(col in data.columns for col in pair):
                data[f"{pair[0]}_x_{pair[1]}"] = data[pair[0]] * data[pair[1]]
                logger.debug(f"Interaction feature {pair[0]}_x_{pair[1]} created.")

    # Model-Based Feature Selection
    if fe_config["feature_selection"]["enabled"] and fe_config["feature_selection"]["model_based"]["enabled"]:
        if "target" in data.columns:
            target_column = "target"
        elif "target_column" in data.columns:
            target_column = "target_column"
        else:
            target_column = None

        if target_column:
            try:
                if ptypes.is_numeric_dtype(data[target_column]):
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(data.drop(target_column, axis=1), data[target_column])
                sfm = SelectFromModel(model, max_features=fe_config["feature_selection"]["model_based"]["num_features"])
                sfm.fit(data.drop(target_column, axis=1), data[target_column])
                selected_features = data.columns[sfm.get_support()]
                data = data[list(selected_features) + [target_column]]
                logger.debug(f"Model-based feature selection applied. Selected features: {selected_features}")
            except Exception as e:
                logger.error(f"Error during model-based feature selection: {e}")

    return data
