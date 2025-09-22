import pandas as pd


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
    skewness['Exceeding_Threshold'] = skewness['Skewness'] >= 1
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
                            columns=['feature', 'n_outliers', 'outlier_pct']).sort_values('outlier_pct', ascending=False) 
    return outlier_df


