from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


class preprocessor:
    """
    Preprocessing utility that handles data loading, cleaning (outside pipeline),
    and builds a scikit-learn ColumnTransformer for consistent transformations.
    """

    def __init__(self, test_size=0.15, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self._column_transformer = None  # will be built during run_preprocessing

    def load_data(self, filepath):
        """Load CSV dataset."""
        return pd.read_csv(filepath)

    def split_data(self, df, target_col):
        """Split into train/validation sets with stratification."""
        X = df.drop(columns=target_col)
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        return X_train, X_val, y_train, y_val

    # ------------------------------------------------------------------
    # Pre-pipeline cleaning steps (row-level operations)
    # ------------------------------------------------------------------
    def resting_bp_preprocessor(self, resting_bp_col, X_train, X_val, y_train, y_val):
        """Remove rows where RestingBP == 0."""
        mask_t = X_train[resting_bp_col] != 0
        mask_v = X_val[resting_bp_col] != 0

        X_train = X_train[mask_t].reset_index(drop=True)
        y_train = y_train[mask_t].reset_index(drop=True)
        X_val = X_val[mask_v].reset_index(drop=True)
        y_val = y_val[mask_v].reset_index(drop=True)
        return X_train, X_val, y_train, y_val

    def oldpeak_preprocessor(self, oldpeak_col, X_train, X_val):
        """Clip negative Oldpeak values to 0."""
        X_train[oldpeak_col] = X_train[oldpeak_col].clip(lower=0)
        X_val[oldpeak_col] = X_val[oldpeak_col].clip(lower=0)
        return X_train, X_val

    def cholesterol_imputer(self, cholesterol_col, X_train, X_val):
        """Convert zero cholesterol to NaN (imputation will happen later in the pipeline)."""
        X_train[cholesterol_col] = X_train[cholesterol_col].replace(0, np.nan)
        X_val[cholesterol_col] = X_val[cholesterol_col].replace(0, np.nan)
        return X_train, X_val

    # ------------------------------------------------------------------
    # Pipeline builder
    # ------------------------------------------------------------------
    def build_preprocessor(self, con_num_features, bin_features,
                           nom_features, ord_features, ord_categories):
        """
        Create the ColumnTransformer that scales, imputes and encodes.
        Returns the transformer (not fitted).
        """
        # Continuous numeric pipeline
        con_num_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Binary categorical
        bin_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oe_encoder", OrdinalEncoder())
        ])

        # Nominal categorical
        nom_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        # Ordinal categorical
        ord_Pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oe_encoder", OrdinalEncoder(categories=[ord_categories]))
        ])

        # ColumnTransformer
        preprocessor = ColumnTransformer([
            ("con_num", con_num_Pipeline, con_num_features),
            ("bin", bin_Pipeline, bin_features),
            ("nom", nom_Pipeline, nom_features),
            ("ord", ord_Pipeline, ord_features)
        ], remainder="drop")

        return preprocessor

    # ------------------------------------------------------------------
    # Full run – uses the pipeline after cleaning
    # ------------------------------------------------------------------
    def run_preprocessing(self, filepath, target_col, resting_bp_col,
                           oldpeak_col, cholesterol_col, continuous_num_features,
                           bin_features, nom_features, ord_features, ord_categories):
        """
        Execute the entire preprocessing workflow.
        Steps:
        1. Load & split
        2. Clean invalid values (RestingBP, Oldpeak, Cholesterol)
        3. Build & apply the preprocessing ColumnTransformer
        4. Return final DataFrames (with feature names) and labels
        """
        # 1. Load and split
        data = self.load_data(filepath)
        X_train, X_val, y_train, y_val = self.split_data(data, target_col)

        # 2. Clean
        X_train, X_val, y_train, y_val = self.resting_bp_preprocessor(
            resting_bp_col, X_train, X_val, y_train, y_val
        )
        X_train, X_val = self.oldpeak_preprocessor(oldpeak_col, X_train, X_val)
        X_train, X_val = self.cholesterol_imputer(cholesterol_col, X_train, X_val)

        # 3. Build transformer
        self._column_transformer = self.build_preprocessor(
            continuous_num_features, bin_features,
            nom_features, ord_features, ord_categories
        )

        # Fit on train, transform both
        X_train_processed = self._column_transformer.fit_transform(X_train)
        X_val_processed = self._column_transformer.transform(X_val)

        # 4. Convert back to DataFrames
        feature_names = self._column_transformer.get_feature_names_out()
        final_X_train = pd.DataFrame(X_train_processed, columns=feature_names)
        final_X_val = pd.DataFrame(X_val_processed, columns=feature_names)

        return final_X_train, final_X_val, y_train, y_val