"""
Preprocessing module for Heart Disease Prediction (ANN).

Provides the preprocessor class that handles:
- Data loading and splitting
- Domain-specific medical cleaning (row-level RestingBP removal,
  column-level Oldpeak clipping, Cholesterol zero → NaN conversion)
- Building a full scikit-learn preprocessing pipeline (column cleaning +
  feature engineering with MinMax scaling, imputation, encoding)
- Applying the pipeline to training and validation sets
"""

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from src_utils import MedicalColumnCleaner


class preprocessor:
    """
    Handles all data preparation steps for the heart disease classification task
    using artificial neural networks. Uses MinMaxScaler for continuous features.

    Parameters
    ----------
    test_size : float, default=0.15
        Proportion of the dataset to use as validation.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, test_size=0.15, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self._full_preprocessor = None

    def load_data(self, filepath):
        """Load CSV dataset. (See ML version for full docstring.)"""
        return pd.read_csv(filepath)

    def split_data(self, df, target_col):
        """Split into stratified train/validation sets. (See ML version.)"""
        X = df.drop(columns=target_col)
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_val, y_train, y_val

    def drop_invalid_resting_bp(self, X, y, resting_bp_col='RestingBP'):
        """Remove rows with RestingBP == 0. (See ML version.)"""
        mask = X[resting_bp_col] != 0
        return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    def build_preprocessor(self, con_num_features, bin_features,
                           nom_features, ord_features, ord_categories,
                           oldpeak_col='Oldpeak', cholesterol_col='Cholesterol'):
        """Build full preprocessing pipeline with MinMaxScaler."""
        con_num_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())          # ← ANN uses MinMaxScaler
        ])

        bin_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oe_encoder", OrdinalEncoder())
        ])

        nom_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        ord_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oe_encoder", OrdinalEncoder(categories=[ord_categories]))
        ])

        column_transformer = ColumnTransformer([
            ("con_num", con_num_Pipeline, con_num_features),
            ("bin", bin_Pipeline, bin_features),
            ("nom", nom_Pipeline, nom_features),
            ("ord", ord_Pipeline, ord_features)
        ], remainder="drop")

        full_preprocessor = Pipeline([
            ("clean_columns", MedicalColumnCleaner(oldpeak_col=oldpeak_col,
                                                   cholesterol_col=cholesterol_col)),
            ("feature_engineering", column_transformer)
        ])

        return full_preprocessor

    def run_preprocessing(self, filepath, target_col, resting_bp_col,
                           oldpeak_col, cholesterol_col, continuous_num_features,
                           bin_features, nom_features, ord_features, ord_categories):
        """Execute full preprocessing and return NumPy arrays."""
        data = self.load_data(filepath)
        X_train, X_val, y_train, y_val = self.split_data(data, target_col)

        X_train, y_train = self.drop_invalid_resting_bp(X_train, y_train, resting_bp_col)
        X_val, y_val = self.drop_invalid_resting_bp(X_val, y_val, resting_bp_col)

        self._full_preprocessor = self.build_preprocessor(
            continuous_num_features, bin_features, nom_features,
            ord_features, ord_categories, oldpeak_col, cholesterol_col
        )

        # For ANN we return NumPy arrays directly (training.py expects them)
        final_X_train = self._full_preprocessor.fit_transform(X_train)
        final_X_val = self._full_preprocessor.transform(X_val)

        return final_X_train, final_X_val, y_train.to_numpy(), y_val.to_numpy()