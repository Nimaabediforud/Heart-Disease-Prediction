from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

#----------------
# Check skewness
#----------------
def skewness_detector(num_cols):
    # Compute skewness for each numerical features and add to dict
    sk = {col: abs(num_cols[col].skew()) for col in num_cols}
    # Convert to dataframe
    skewness = pd.DataFrame(sk, index=[0])
    # Melt (for better inspection)
    skewness = pd.melt(skewness, var_name='Feature', value_name='Skewness')
    # Check if skewness is exceeding the threshold (skewness >= 1)
    skewness['Skewness_Exceeding_Threshold'] = skewness['Skewness'] >= 1
    return skewness


#-------------------------------------------
# Find potential outliers in numeric values
#-------------------------------------------
def outlier_detector(num_cols):
    # IQR method (1.5 * IQR) 
    outlier_info = [] 
    # Loop through numeric features
    for c in num_cols: 
        # Calculate Q1 -> 25%
        Q1 = num_cols[c].quantile(0.25) 
        # Calculate Q3 -> 75%
        Q3 = num_cols[c].quantile(0.75) 
        # Calculate IQR
        IQR = Q3 - Q1 
        # Calculate and determine lower and upper bounds
        lower = Q1 - 1.5 * IQR 
        upper = Q3 + 1.5 * IQR 
        # Filter outliers
        mask = (num_cols[c] < lower) | (num_cols[c] > upper) 
        # Count outliers
        n_out = mask.sum() 
        # Add all info to list (features, number of outliers, percentage of outliers)
        outlier_info.append((c, n_out, (n_out / len(num_cols) * 100).round(3)))

    # Convert to dataframe
    outlier_df = pd.DataFrame(outlier_info,
                           columns=['Feature', 'N_outliers', 'Outlier_pct']).sort_values(by="Outlier_pct", ascending=False)
    return outlier_df


#-------------------------------------------
# Drop invalid RestingBP values (=0)
#-------------------------------------------
def drop_invalid_resting_bp(X, y):
    # Set mask: Keep all rows except the one that's 0
    mask = X['RestingBP'] != 0
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)



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
    
