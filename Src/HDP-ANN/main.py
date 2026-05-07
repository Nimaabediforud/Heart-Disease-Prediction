from .preprocessing import preprocessor
from .training import trainer

# 1. Preprocessing
prep = preprocessor(test_size=0.15, random_state=42)
data = prep.load_data("data/heart.csv")
X_train, X_val, y_train, y_val = prep.split_data(data, "HeartDisease")

X_train, X_val, y_train, y_val = prep.resting_bp_preprocessor(
    "RestingBP", X_train, X_val, y_train, y_val)
X_train, X_val = prep.oldpeak_preprocessor("Oldpeak", X_train, X_val)
X_train, X_val = prep.cholesterol_imputer("Cholesterol", X_train, X_val)

# 2. Build and fit the ColumnTransformer (now with MinMaxScaler)
column_trans = prep.build_preprocessor(
    con_num_features=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
    bin_features=['Sex', 'ExerciseAngina'],
    nom_features=['ChestPainType', 'RestingECG'],
    ord_features=['ST_Slope'],
    ord_categories=['Down', 'Flat', 'Up']
)
column_trans.fit(X_train)

# Transform to NumPy arrays
final_X_train = column_trans.transform(X_train)
final_X_val = column_trans.transform(X_val)

# 3. Train ANN
tr = trainer(dropout=0.3, decision_threshold=0.45)
model, history = tr.train(final_X_train, final_X_val, y_train.to_numpy(), y_val.to_numpy())

# 4. Save and evaluate
model.save("models/ann_model.keras")
metrics = tr.evaluate("models/ann_model.keras", final_X_val, y_val.to_numpy())