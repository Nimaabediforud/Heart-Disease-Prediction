# REG-ANN-EXT2.ipynb -> results of model selection, STAGE 2 
stage2_results = {
    "A1": {
        "Smoker": {
            "R2": 0.6361,
            "MAE": 3627.81,
            "RMSE": 7105.10,
            "Epochs": 30
        },
        "BMI/Age": {
            "R2": 0.5815,
            "MAE": 4056.09,
            "RMSE": 7619.67,
            "Epochs": 30
        },
        "All": {
            "R2": 0.7092,
            "MAE": 3351.39,
            "RMSE": 6351.83,
            "Epochs": 30
        }
    },

    "A2": {
        "Smoker": {
            "R2": 0.8102,
            "MAE": 2767.51,
            "RMSE": 5131.74,
            "Epochs": 30
        },
        "BMI/Age": {
            "R2": 0.5816,
            "MAE": 4100.27,
            "RMSE": 7618.58,
            "Epochs": 14
        },
        "All": {
            "R2": 0.7591,
            "MAE": 3191.89,
            "RMSE": 5781.07,
            "Epochs": 22
        }
    },

    "A3": {
        "Smoker": {
            "R2": 0.7526,
            "MAE": 3158.61,
            "RMSE": 5857.97,
            "Epochs": 20
        },
        "BMI/Age": {
            "R2": 0.5610,
            "MAE": 3985.33,
            "RMSE": 7803.49,
            "Epochs": 20
        },
        "All": {
            "R2": 0.7139,
            "MAE": 3326.94,
            "RMSE": 6300.33,
            "Epochs": 20
        }
    },

    "A4": {
        "Smoker": {
            "R2": 0.6591,
            "MAE": 3968.25,
            "RMSE": 6877.21,
            "Epochs": 20
        },
        "BMI/Age": {
            "R2": 0.4740,
            "MAE": 4551.92,
            "RMSE": 8542.08,
            "Epochs": 13
        },
        "All": {
            "R2": 0.4671,
            "MAE": 4522.23,
            "RMSE": 8597.83,
            "Epochs": 13
        }
    }
}


# REG-ANN-EXT2.ipynb & REG-ML-EXT2.ipynb -> results of model evaluation on the test set for both ML and ANN
comparison = {
    "Model": [
        "Ridge Regression",
        "Artificial Neural Network"
    ],
    "R²": [
        0.8825,
        0.8425
    ],
    "MAE": [
        2298.73,
        2278.55
    ],
    "RMSE": [
        4230.34,
        4897.56
    ]
}

