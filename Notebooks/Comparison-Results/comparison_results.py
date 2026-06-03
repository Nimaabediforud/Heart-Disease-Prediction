
# HeartDisease prediction
classification_results = {
    "Stacking": {
        "Accuracy": 0.90,
        "F1": 0.8982,
        "FN": 5,
        "FP": 9,
        "Threshold": 0.40
    },

    "ANN": {
        "Accuracy": 0.88,
        "F1": 0.8839,
        "FN": 7,
        "FP": 9,
        "Threshold": 0.50
    }
}

# Biomedical insurance cost prediction
regression_results = {
    "Ridge": {
        "R2": 0.8825,
        "MAE": 2298.73,
        "RMSE": 4230.34
    },

    "ANN": {
        "R2": 0.8425,
        "MAE": 2278.55,
        "RMSE": 4897.56
    }
}

# ANN regression evolution
ann_reg_evolution = {
    "Raw Target\nA": -0.0523,
    "Log Target\nA": 0.6046,
    "A1": 0.6361,
    "A2": 0.8102,
    "A3": 0.7526,
    "A4": 0.6591,
    "Stage3\nBest": 0.8626,
    "Final Test": 0.8425
}

# ML (Ridge) regression evolution
ridge_evolution = {
    "Stage": [
        "Raw Target\n(Val)",
        "Log Target\n(Val)",
        "Final Test"
    ],
    "R2": [
        0.8639,
        0.8184,
        0.8825
    ]
}

# Project evolution
project_evolution = {
    "Step": [
        "Heart Disease Classification (HD UCI Dataset)",
        "ANN Classification (HD UCI Dataset)",
        "Regression Attempt-Cholesterol (HD UCI Dataset)",
        "Regression Attempt-Oldpeak (HD UCI Dataset)",
        "Regression Attempt-Cholesterol (Framingham Dataset)",
        "ML Regression-insurance-cost (Insurance Dataset)",
        "ANN Regression-insurance-cost (Insurance Dataset)"
    ]
}
