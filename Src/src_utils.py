from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


#-----------------------------------------------------------
# Custom cleanup operation considering medical constraints
#-----------------------------------------------------------
class MedicalColumnCleaner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that applies domain-specific
    cleaning to certain columns:
    - Clips negative Oldpeak values to 0.
    - Converts Cholesterol values of 0 to NaN (imputation will be done later by the pipeline).
    """
    def __init__(self, oldpeak_col='Oldpeak', cholesterol_col='Cholesterol'):
        self.oldpeak_col = oldpeak_col
        self.cholesterol_col = cholesterol_col

    def fit(self, X, y=None):
        return self  # no fitting required

    def transform(self, X):
        X = X.copy()
        X[self.oldpeak_col] = X[self.oldpeak_col].clip(lower=0)
        X[self.cholesterol_col] = X[self.cholesterol_col].replace(0, np.nan)
        return X
    

