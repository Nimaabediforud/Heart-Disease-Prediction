"""
Preprocessing module for Heart Disease Prediction (ML).

Provides the preprocessor class that handles:
- Data loading and splitting
- Domain-specific medical cleaning (row-level RestingBP removal,
  column-level Oldpeak clipping, Cholesterol zero → NaN conversion)
- Building a full scikit-learn preprocessing pipeline (column cleaning +
  feature engineering with scaling, imputation, encoding)
- Applying the pipeline to training and validation sets
"""

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from src_utils import MedicalColumnCleaner


class preprocessor:
    """
    Handles all data preparation steps for the heart disease classification task.

    Parameters
    ----------
    test_size : float, default=0.15
        Proportion of the dataset to use as validation.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, test_size=0.15, random_state=42):
        """Initialize with split parameters."""
        self.test_size = test_size
        self.random_state = random_state
        self._full_preprocessor = None   # Stores the fitted preprocessing pipeline

    def load_data(self, filepath):
        """
        Load the dataset from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        return pd.read_csv(filepath)

    def split_data(self, df, target_col):
        """
        Split the dataset into training and validation sets (stratified).

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset.
        target_col : str
            Name of the target column.

        Returns
        -------
        tuple : (X_train, X_val, y_train, y_val)
            Feature and label DataFrames/Series.
        """
        X = df.drop(columns=target_col)
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_val, y_train, y_val

    def drop_invalid_resting_bp(self, X, y, resting_bp_col='RestingBP'):
        """
        Remove rows where RestingBP equals 0 (biologically impossible).

        Because this affects both features and labels, it must be done
        outside the scikit-learn pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.
        y : pd.Series
            Target labels.
        resting_bp_col : str, default='RestingBP'
            Name of the resting blood pressure column.

        Returns
        -------
        tuple : (X_clean, y_clean)
            Filtered features and labels.
        """
        mask = X[resting_bp_col] != 0
        return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    def build_preprocessor(self, con_num_features, bin_features,
                           nom_features, ord_features, ord_categories,
                           oldpeak_col='Oldpeak', cholesterol_col='Cholesterol'):
        """
        Construct the full preprocessing pipeline.

        Steps:
        1. MedicalColumnCleaner – clips Oldpeak and converts cholesterol zeros to NaN.
        2. ColumnTransformer – imputes, scales, and encodes all features.

        Parameters
        ----------
        con_num_features : list of str
            Continuous numerical feature names.
        bin_features : list of str
            Binary categorical feature names.
        nom_features : list of str
            Nominal categorical feature names.
        ord_features : list of str
            Ordinal categorical feature names.
        ord_categories : list of list
            Ordered categories for each ordinal feature.
        oldpeak_col : str, default='Oldpeak'
            Name of the Oldpeak column.
        cholesterol_col : str, default='Cholesterol'
            Name of the Cholesterol column.

        Returns
        -------
        full_preprocessor : sklearn.pipeline.Pipeline
            Unfitted pipeline (cleaner + feature engineering).
        """
        # Sub-pipeline for continuous numeric features
        con_num_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Sub-pipeline for binary categorical features
        bin_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oe_encoder", OrdinalEncoder())
        ])

        # Sub-pipeline for nominal categorical features
        nom_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        # Sub-pipeline for ordinal categorical features
        ord_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oe_encoder", OrdinalEncoder(categories=[ord_categories]))
        ])

        # Column transformer combining all feature pipelines
        column_transformer = ColumnTransformer([
            ("con_num", con_num_Pipeline, con_num_features),
            ("bin", bin_Pipeline, bin_features),
            ("nom", nom_Pipeline, nom_features),
            ("ord", ord_Pipeline, ord_features)
        ], remainder="drop")

        # Full pipeline: clean columns first, then apply feature engineering
        full_preprocessor = Pipeline([
            ("clean_columns", MedicalColumnCleaner(oldpeak_col=oldpeak_col,
                                                   cholesterol_col=cholesterol_col)),
            ("feature_engineering", column_transformer)
        ])

        return full_preprocessor

    def run_preprocessing(self, filepath, target_col, resting_bp_col,
                           oldpeak_col, cholesterol_col, continuous_num_features,
                           bin_features, nom_features, ord_features, ord_categories):
        """
        Execute the entire preprocessing workflow and return final arrays.

        Workflow:
        1. Load and split the data.
        2. Remove invalid RestingBP rows (outside pipeline).
        3. Build the full preprocessing pipeline.
        4. Fit on training data, transform training and validation sets.
        5. Convert results to DataFrames with feature names.

        Parameters
        ----------
        filepath : str
            Path to the CSV dataset.
        target_col : str
            Name of the target column.
        resting_bp_col : str
            Column name for RestingBP.
        oldpeak_col : str
            Column name for Oldpeak.
        cholesterol_col : str
            Column name for Cholesterol.
        continuous_num_features : list of str
            Continuous numeric features to scale.
        bin_features : list of str
            Binary categorical features.
        nom_features : list of str
            Nominal categorical features.
        ord_features : list of str
            Ordinal categorical features.
        ord_categories : list of list
            Order of categories for ordinal features.

        Returns
        -------
        final_X_train : pd.DataFrame
            Preprocessed training features.
        final_X_val : pd.DataFrame
            Preprocessed validation features.
        y_train : pd.Series
            Training labels.
        y_val : pd.Series
            Validation labels.
        """
        # 1. Load & split
        data = self.load_data(filepath)
        X_train, X_val, y_train, y_val = self.split_data(data, target_col)

        # 2. Row-level cleaning (RestingBP)
        X_train, y_train = self.drop_invalid_resting_bp(X_train, y_train, resting_bp_col)
        X_val, y_val = self.drop_invalid_resting_bp(X_val, y_val, resting_bp_col)

        # 3. Build the full preprocessing pipeline
        self._full_preprocessor = self.build_preprocessor(
            continuous_num_features, bin_features, nom_features,
            ord_features, ord_categories, oldpeak_col, cholesterol_col
        )

        # 4. Fit on train, transform both
        final_X_train = self._full_preprocessor.fit_transform(X_train)
        final_X_val = self._full_preprocessor.transform(X_val)

        # 5. Create DataFrames with proper feature names (optional but nice)
        feature_names = self._full_preprocessor.get_feature_names_out()
        final_X_train = pd.DataFrame(final_X_train, columns=feature_names)
        final_X_val = pd.DataFrame(final_X_val, columns=feature_names)

        return final_X_train, final_X_val, y_train, y_val