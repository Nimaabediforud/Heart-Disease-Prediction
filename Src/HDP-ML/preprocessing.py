#----- Tools, visualizers -----
import pandas as pd
import numpy as np

#----- Preprocessing -----
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler



class preprocessor:
    """
    A preprocessing utility class that provides functions for loading,
    cleaning, transforming, and encoding data for the Heart Disease Predictor project.
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor with default split parameters.
        
        Args:
            test_size (float): Proportion of validation set.
            random_state (int): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state


    def load_data(self, filepath):
        """
        Load dataset from a CSV file into a pandas DataFrame.
        
        Args:
            filepath (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        return pd.read_csv(filepath)
    

    def split_data(self, df, target_col):
        """
        Split dataset into training and validation sets.
        
        Args:
            df (pd.DataFrame): Full dataset.
            target_col (str): Name of the target column.
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        # Separate features and labels cols
        X = df.drop(columns=target_col) # Features
        y = df[target_col] # Labels

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15 , random_state=self.random_state, stratify=y
        )
        return (X_train, X_val, y_train, y_val)
    

    def resting_bp_preprocessor(self, resting_bp_col, X_train, X_val, y_train, y_val):
        """
        Remove rows where resting blood pressure is zero, as they are invalid.
        
        Args:
            resting_bp_col (str): Column name for resting blood pressure.
            X_train, X_val (pd.DataFrame): Features for training and validation.
            y_train, y_val (pd.Series): Labels for training and validation.
        
        Returns:
            tuple: Filtered (X_train, X_val, y_train, y_val)
        """
        # Set masks: Keep all rows except the one that's 0
        mask_t = X_train[resting_bp_col] != 0
        mask_v = X_val[resting_bp_col] != 0

        #  Train
        #--------------
        # Filter X_train
        X_train = X_train[mask_t].reset_index(drop=True)

        # Filter y_train
        y_train = y_train[mask_t].reset_index(drop=True)

        #  Validation
        #---------------
        # Filter X_val
        X_val = X_val[mask_v].reset_index(drop=True)

        # Filter y_val
        y_val = y_val[mask_v].reset_index(drop=True)

        return (X_train, X_val, y_train, y_val)
    

    def oldpeak_preprocessor(self, oldpeak_col, X_train, X_val):
        # Filter X_train
        X_train[oldpeak_col] = X_train[oldpeak_col].apply(lambda x: 0 if x < 0 else x)

        # Filter X_val
        X_val[oldpeak_col] = X_val[oldpeak_col].apply(lambda x: 0 if x < 0 else x)

        return (X_train, X_val)
    

    def cholesterol_imputer(self, choleterol_col, X_train, X_val):
        """
        Replace negative values in 'Oldpeak' column with 0.
        
        Args:
            oldpeak_col (str): Column name for oldpeak values.
            X_train, X_val (pd.DataFrame): Features for training and validation.
        
        Returns:
            tuple: Updated (X_train, X_val)
        """
        # Convert zeros into Nan
        X_train[choleterol_col] = X_train[choleterol_col].replace(0, np.nan)
        X_val[choleterol_col] = X_val[choleterol_col].replace(0, np.nan)

        # Define median imputer
        median_imputer = SimpleImputer(strategy='median')

        # Apply on X_train
        X_train[choleterol_col] = median_imputer.fit_transform(X_train[[choleterol_col]])

        # Apply on X_val
        X_val[choleterol_col] = median_imputer.transform(X_val[[choleterol_col]])

        return (X_train, X_val)


    def feature_scaler(self, continuous_num_features, X_train, X_val):
        """
        Impute missing or zero cholesterol values with the median.
        
        Args:
            choleterol_col (str): Column name for cholesterol.
            X_train, X_val (pd.DataFrame): Features for training and validation.
        
        Returns:
            tuple: Updated (X_train, X_val) with imputed cholesterol.
        """
        # Define StandardScaler
        std_scaler = StandardScaler()

        #  Train
        #--------------
        X_train_scaled = pd.DataFrame(
            std_scaler.fit_transform(X_train[continuous_num_features]),
            columns=continuous_num_features,
            index=X_train.index
            )

        #  Validation
        #--------------
        X_val_scaled = pd.DataFrame(
            std_scaler.transform(X_val[continuous_num_features]),
            columns=continuous_num_features,
            index=X_val.index
            )

        return (X_train_scaled, X_val_scaled)


    def cat_features_encoder(self, bin_features, nom_features,
                              ord_features, ord_categories,
                               X_train_scaled, X_val_scaled, X_train, X_val):
        """
        Encode categorical features:
        - Binary features: Label Encoding
        - Nominal features: One-Hot Encoding
        - Ordinal features: Ordinal Encoding
        
        Args:
            bin_features (list): Binary feature column names.
            nom_features (list): Nominal feature column names.
            ord_features (list): Ordinal feature column names.
            ord_categories (list of lists): Categories for each ordinal feature.
            X_train_scaled, X_val_scaled (pd.DataFrame): Scaled continuous features.
            X_train, X_val (pd.DataFrame): Original training and validation data.
        
        Returns:
            tuple: (final_X_train, final_X_val) with all encoded and scaled features.
        """
        # Define encoders
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ore = OrdinalEncoder(categories=ord_categories) 

        #-------------
        # Train
        #-------------

        # Label encoding 
        X_train_bin_encoded = pd.DataFrame(
            {col: LabelEncoder().fit_transform(X_train[col]) for col in bin_features},
            index=X_train.index
        )

        # One-hot encoding
        X_train_nom_encoded = pd.DataFrame(
            ohe.fit_transform(X_train[nom_features]),
            columns=ohe.get_feature_names_out(nom_features),
            index=X_train.index
        )

        # Ordinal encoding 
        X_train_ord_encoded = pd.DataFrame(
            ore.fit_transform(X_train[ord_features]),
            columns=ord_features,
            index=X_train.index
        )

        #-------------
        # Validation
        #-------------

        # Label encoding 
        X_val_bin_encoded = pd.DataFrame(
            {col: LabelEncoder().fit(X_train[col]).transform(X_val[col]) for col in bin_features},
            index=X_val.index
        )

        # One-hot encoding
        X_val_nom_encoded = pd.DataFrame(
            ohe.transform(X_val[nom_features]),
            columns=ohe.get_feature_names_out(nom_features),
            index=X_val.index
        )

        # Ordinal encoding 
        X_val_ord_encoded = pd.DataFrame(
            ore.transform(X_val[ord_features]),
            columns=ord_features,
            index=X_val.index
        )
    
        #-------------
        # Concatenate
        #-------------

        final_X_train = pd.concat([
            X_train['FastingBS'],
            X_train_bin_encoded,
            X_train_nom_encoded, 
            X_train_ord_encoded,
            X_train_scaled], axis=1)


        final_X_val = pd.concat([
            X_val['FastingBS'],
            X_val_bin_encoded,
            X_val_nom_encoded, 
            X_val_ord_encoded,
            X_val_scaled], axis=1)

        return (final_X_train, final_X_val)
    


