"""
Utilities for encoding annotator metadata as one-hot/categorical features.
This enables better correlation analysis and interpretability.

@author DisCo Authors (Enhanced)
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import json

def extract_categorical_features(metadata_dict):
    """
    Extract categorical features from metadata dictionary.

    Args:
        metadata_dict: Dictionary mapping annotator_id -> metadata dict
                      Metadata dict contains keys like 'Age', 'Gender', 'Nationality', etc.

    Returns:
        feature_names: List of feature names
        categorical_values: Dict mapping feature_name -> list of unique values
        feature_dimensions: Dict mapping feature_name -> number of unique values
    """
    feature_values = defaultdict(set)

    for ann_id, metadata in metadata_dict.items():
        for key, value in metadata.items():
            if value and value not in ['unknown', 'DATA_EXPIRED', 'unknown age',
                                      'unknown gender', 'unknown nationality',
                                      'unknown country']:
                feature_values[key].add(str(value))

    # Convert sets to sorted lists for consistent ordering
    categorical_values = {}
    feature_dimensions = {}
    feature_names = sorted(feature_values.keys())

    for feature_name in feature_names:
        values = sorted(list(feature_values[feature_name]))
        categorical_values[feature_name] = values
        feature_dimensions[feature_name] = len(values)

    return feature_names, categorical_values, feature_dimensions


def encode_metadata_onehot(metadata_dict, categorical_values, feature_names):
    """
    Encode metadata as one-hot vectors.

    Args:
        metadata_dict: Dictionary mapping annotator_id -> metadata dict
        categorical_values: Dict mapping feature_name -> list of unique values
        feature_names: List of feature names to encode

    Returns:
        encoded_dict: Dictionary mapping annotator_id -> one-hot vector (numpy array)
        total_dim: Total dimension of one-hot encoding
    """
    # Calculate total dimension
    total_dim = sum(len(categorical_values[fname]) for fname in feature_names)

    encoded_dict = {}

    for ann_id, metadata in metadata_dict.items():
        onehot_vector = np.zeros(total_dim, dtype=np.float32)
        offset = 0

        for feature_name in feature_names:
            value = metadata.get(feature_name, None)
            if value and str(value) in categorical_values[feature_name]:
                idx = categorical_values[feature_name].index(str(value))
                onehot_vector[offset + idx] = 1.0
            # If value is missing or unknown, leave as zeros (no one-hot encoding)

            offset += len(categorical_values[feature_name])

        encoded_dict[ann_id] = onehot_vector

    return encoded_dict, total_dim


def save_metadata_encoding_config(feature_names, categorical_values, feature_dimensions,
                                 output_path):
    """Save metadata encoding configuration to file."""
    config = {
        'feature_names': feature_names,
        'categorical_values': {k: list(v) for k, v in categorical_values.items()},
        'feature_dimensions': feature_dimensions
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_metadata_encoding_config(config_path):
    """Load metadata encoding configuration from file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    categorical_values = {k: list(v) for k, v in config['categorical_values'].items()}

    return config['feature_names'], categorical_values, config['feature_dimensions']
