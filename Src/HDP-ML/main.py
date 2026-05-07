from .preprocessing import preprocessor
from .training import trainer

# 1. Initialise & load
prep = preprocessor(test_size=0.15, random_state=42)
data = prep.load_data("data/heart.csv")                     # adjust path
X_train, X_val, y_train, y_val = prep.split_data(data, "HeartDisease")

# 2. Manual cleaning steps (before pipeline)
X_train, X_val, y_train, y_val = prep.resting_bp_preprocessor(
    "RestingBP", X_train, X_val, y_train, y_val)
X_train, X_val = prep.oldpeak_preprocessor("Oldpeak", X_train, X_val)
X_train, X_val = prep.cholesterol_imputer("Cholesterol", X_train, X_val)

# 3. Build & fit the ColumnTransformer
column_trans = prep.build_preprocessor(
    con_num_features=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
    bin_features=['Sex', 'ExerciseAngina'],
    nom_features=['ChestPainType', 'RestingECG'],
    ord_features=['ST_Slope'],
    ord_categories=['Down', 'Flat', 'Up']
)
column_trans.fit(X_train)

# 4. Train final model
tr = trainer(random_state=42)
final_pipeline = tr.train(column_trans, X_train, y_train,
                          save_path="models/final_model.joblib")

# 5. Evaluate
metrics = tr.evaluate(final_pipeline, X_val, y_val)
