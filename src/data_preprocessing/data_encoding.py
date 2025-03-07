import pandas as pd
import logging
import yaml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_categorical_data(data, logger):
    """Encodes categorical data using config parameters."""
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    encoding_config = config["feature_engineering"]["encoding"]
    categorical_threshold = encoding_config["categorical_threshold"]
    default_encoding = encoding_config["default_encoding"]

    categorical_cols = data.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if data[col].nunique() <= categorical_threshold:
            if default_encoding == "onehot":
                logger.debug(f"One-hot encoding column: {col}")
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_cols = encoder.fit_transform(data[[col]])
                encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([col]))
                data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)
            elif default_encoding == "label":
                logger.debug(f"Label encoding column: {col}")
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
        else:
            logger.debug(f"Skipping encoding for column: {col} due to high cardinality.")

    return data
