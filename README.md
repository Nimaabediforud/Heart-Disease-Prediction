# Heart Disease Prediction Project (ML & ANN)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-blue?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-success?logo=pandas)
![License](https://img.shields.io/github/license/Nimaabediforud/Heart-Disease-Prediction)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

---

## Overview / Objective

This project focuses on predicting whether a patient has heart disease based on clinical and demographic features.
The project covers:
- Preprocessing data for both machine learning and artificial neural network models.
- Scaling continuous features and encoding categorical variables.
- Training and evaluating multiple traditional machine learning models (Random Forest, Gradient Boosting, AdaBoost, Extra Trees, and Stacking Classifier).
- Building and training an artificial neural network with hyperparameter tuning and decision threshold adjustments.
- Comparing model performance using metrics such as F1-score, accuracy, confusion matrix, and classification reports.

The project highlights **the end-to-end workflow of data preprocessing, model building, evaluation, and comparison** for a healthcare prediction task.

## Project Structure

The project is organized as follows:
```
Heart-Disease-Prediction/
│
├── Data/
│ └── dataset.csv   # Original heart disease dataset
│
├── Models/
│ └── trained models # Saved machine learning and ANN 
│
├── Notebooks/
│ ├── EDA.ipynb   # Exploratory Data Analysis
│ ├── HDP-ML.ipynb   # Traditional Machine Learning workflow
│ ├── HDP-ANN.ipynb   # Artificial Neural Network workflow
│ └── utils.py   # Utility functions used across notebooks
│
├── Src/
│ ├── HDP-ML/
│ │ ├── init.py
│ │ ├── preprocessing.py
│ │ └── training.py
│ └── HDP-ANN/
│ ├── init.py
│ ├── preprocessing.py
│ └── training.py
│
├── README.md   # Project documentation
├── requirements.txt   # Project dependencies
└── LICENSE   # License for the project
```
## Data Description

The project uses a dataset containing patient information and heart disease indicators. The dataset is provided as a CSV file located in the `data/` folder. It includes both numerical and categorical features that describe patient demographics, medical history, and clinical measurements.  

### Key Features:

- **Numerical Features:**
  - Continuous variables such as `Age`, `RestingBP`, `Cholesterol`, `Oldpeak`, `MaxHR`.
  - These features are scaled using `StandardScaler` & `MinMaxScaler` during preprocessing for neural network training.

- **Categorical Features:**
  - **Binary:** e.g., `Sex`, `ExerciseAngina`, `FastingBS`.
  - **Nominal:** e.g., `ChestPainType`, `RestingECG`.
  - **Ordinal:** e.g., `ST_Slope`, encoded according to predefined category order.

- **Target Feature:**
  - `HeartDisease` (0 = No, 1 = Yes) indicating whether the patient has heart disease.

> Note: Some features require preprocessing to handle missing or invalid values. For example, `RestingBP` values of 0 are replaced, `Oldpeak` negative values are corrected, and missing `Cholesterol` values are imputed using the median.

## Key Features & Highlights

- **Comprehensive Heart Disease Prediction:**  
  Implements both traditional machine learning models and artificial neural networks to predict the presence of heart disease based on patient features.

- **Data Preprocessing:**  
  - Scaling of continuous features using `StandardScaler` & `MinMaxScaler`.  
  - Encoding of categorical features (binary, nominal, and ordinal).  
  - Handling of missing or incorrect values in key medical features.
  
    > I tried to handle each step `manually` to better understand how different preprocessing techniques work and how they affect the data and the model’s performance. I wanted to experiment with them myself and learn how to manage data transformations and scaling without relying on ready-to-use tools(scikit-learn `Pipeline` / `ColumnTransformer`). I also wrote a couple of custom classes for preprocessing(within `src/HDP-ML/preprocessing.py` and `src/HDP-ANN/preprocessing.py`), just to practice how these processes can be built from scratch in code. Of course, the easiest and most professional way to handle this would be to use **pipelines**; In fact, That's what should be done. I kept it manual to focus more on the learning process and the details behind each step.

- **Machine Learning Models (HDP-ML):**  
  - Stacking Classifier combining Random Forest, Gradient Boosting, AdaBoost, Extra Trees, and SVC.  
  - Trained and evaluated with F1 score, classification report, and confusion matrix metrics.

- **Artificial Neural Networks (HDP-ANN):**  
  - Flexible architecture with multiple hidden layers and dropout for regularization.  
  - Customizable decision threshold to balance sensitivity and specificity.  
  - Early stopping to prevent overfitting during training.

- **Evaluation Metrics:**  
  - F1 Score (weighted)  
  - Accuracy  
  - Confusion Matrix  
  - Detailed classification reports

- **Organized Codebase:**  
  - Clear separation of preprocessing, training, and evaluation code for ML and ANN workflows.  
  - Notebooks for EDA, ML, and ANN experiments.  
  - Reusable utility functions.

## How to Use / Run the Project

1. **Prepare the Dataset:**  
   Place the dataset CSV file in the `data/` folder.

2. **Preprocessing:**  
   Use the preprocessing scripts in `Src/HDP-ML/preprocessing.py` and `Src/HDP-ANN/preprocessing.py` to process the data.  
   This includes scaling continuous features, encoding categorical features, and handling missing or invalid values.

3. **Training the Models:**  
   - For traditional ML models, use the training scripts in `Src/HDP-ML/training.py`.  
   - For neural networks, use the scripts in `Src/HDP-ANN/training.py`.  

4. **Evaluation:**  
   Evaluate models using the provided evaluation functions. Metrics include F1 Score, accuracy, confusion matrix, and classification reports.

5. **Exploration:**  
   For exploratory data analysis, refer to the notebooks in `notebooks/`:
   - `EDA.ipynb` – basic data exploration and visualization.
   - `HDP-ML.ipynb` – ML model experiments.
   - `HDP-ANN.ipynb` – ANN model experiments.

> Note: The project is designed to work with the provided dataset. No additional configuration or installations are strictly required beyond the listed dependencies.

## Conclusion and Results Comparison

In this project, we implemented two approaches to predict whether a patient has heart disease:

1. **Traditional Machine Learning Models (HDP-ML)**
   - Models used: Random Forest, Gradient Boosting, AdaBoost, Extra Trees, SVC, and a Stacking Classifier.
   - The final stacked model achieved a **weighted F1 score of ~89%** on the validation set.
   - These models performed very well despite the relatively small size of the dataset, showing strong generalization capabilities.

2. **Artificial Neural Network (HDP-ANN)**
   - Architecture: Multi-layer feedforward network with two hidden layers (16 units and 8 units) and dropout of 0.3 to prevent overfitting.
   - Training: 30 epochs with early stopping based on validation loss.
   - Evaluation: After converting predicted probabilities to binary labels using a 0.45 decision threshold, the ANN achieved a **weighted F1 score of ~87%**.
   - Confusion matrix analysis helped fine-tune the decision threshold, reducing false negatives while keeping false positives at an acceptable level.

### Key Observations:

- Both approaches yielded strong results, with only a small difference (~2–3%) in F1 scores.
- Traditional ML models slightly outperformed the neural network in this specific case, likely due to the small dataset size and effective feature engineering.
- The neural network provides flexibility for future expansion or integration of additional features or data, while traditional ML remains robust and efficient for smaller datasets.

> Overall, this project demonstrates the effectiveness of both traditional ML models and neural networks in predicting heart disease, highlighting preprocessing, feature engineering, and model evaluation strategies.

## References / Acknowledgments

- **Dataset**: The dataset used in this project is from [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It contains patient health metrics and indicators relevant to heart disease prediction.  
- **Libraries and Tools**: This project leverages Python libraries such as `pandas`, `numpy`, `scipy`, `scikit-learn`, `tensorflow`, `joblib`, `matplotlib`, and `seaborn` for data preprocessing, model training, evaluation, and visualization.  
- **Inspiration & Learning**: The project workflow and techniques were guided by standard practices in machine learning and deep learning, including feature scaling, encoding, stacking classifiers, neural network design, and evaluation strategies.  

