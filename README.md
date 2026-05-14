# Heart Disease Prediction Project (ML & ANN)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-blue?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-success?logo=pandas)
![License](https://img.shields.io/github/license/Nimaabediforud/Heart-Disease-Prediction)
![Status](https://img.shields.io/badge/Status-Ongoing-success?style=flat-square)

---

## Overview / Objective

This project focuses on predicting whether a patient has heart disease based on clinical and demographic features.
It follows a **structured, pipeline‑based methodology** covering:

- Preprocessing data for both machine learning and artificial neural network models using **scikit‑learn pipelines** and a custom transformer for domain‑specific cleaning.
- Scaling continuous features (`StandardScaler` for ML, `MinMaxScaler` for ANN) and encoding categorical variables.
- Benchmarking multiple traditional machine learning models (Random Forest, Gradient Boosting, AdaBoost, Extra Trees, SVC, Logistic Regression, KNN, Naive Bayes, Gaussian Process) and selecting the best performer.
- Building and training an artificial neural network with hyperparameter tuning, early stopping, and a custom decision threshold.
- Comparing model performance using weighted F1‑score, accuracy, confusion matrices, and classification reports.

The project highlights **an end‑to‑end, reproducible workflow for healthcare prediction tasks**, with a strong emphasis on code modularity and educational clarity.
> The current release focuses on the classification paradigm. Regression and unsupervised learning paradigms are under active development and will follow the same pipeline‑based methodology.

---

## Roadmap / Future Work

- [x] Classification (ML & ANN) – complete, with full pipeline and saved models.
- [ ] Regression (ML & ANN) – predict continuous medical variables (e.g., Cholesterol) using the same preprocessing framework.
- [ ] Unsupervised Learning – clustering and pattern discovery without target labels.
- [ ] Extended evaluation across additional medical datasets to test framework generalizability.

---

## Project Structure
```
Heart-Disease-Prediction/
├── Data/
│   └── heart-data.csv
│
├── Models/
│   ├── Classification/     # fully implemented
│   ├── Regression/         # under development
│   └── Unsupervised/       # planned
│
├── Notebooks/
│   ├── EDA/
│   │   └── EDA.ipynb
│   ├── Classification/
│   │   ├── HDP-CLA-ML.ipynb
│   │   └── HDP-CLS-ANN.ipynb
│   ├── Regression/
│   │   ├── CH-REG-ML.ipynb
│   │   └── CH-REG-ANN.ipynb
│   ├── Unsupervised/
│   │
│   └── utils.py                         # shared utilities (outlier/skewness detectors, DataCleaner)
│
├── Src/
│   ├── Classification/
│   │   ├── HDP-ML/
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py
│   │   │   └── training.py
│   │   └── HDP-ANN/
│   │       ├── __init__.py
│   │       ├── preprocessing.py
│   │       └── training.py
│   ├── Regression/                      # (future) regression source code
│   ├── Unsupervised/                    # (future) unsupervised source code
│   └── src_utils.py                     # shared custom transformer (MedicalColumnCleaner)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
---

## Data Description

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


---

## Key Features & Highlights

- **Comprehensive Heart Disease Prediction:**  
  Implements both traditional machine learning models and an artificial neural network using a **unified preprocessing pipeline**.

- **Professional Data Preprocessing:**  
  - **Domain‑specific cleaning** via a custom `MedicalColumnCleaner` transformer that clips impossible values (Oldpeak) and marks missing cholesterol values for imputation.  
  - Row‑level cleaning (removing patients with `RestingBP == 0`) is handled explicitly outside the pipeline because it affects both features and labels.  
  - Scaling of continuous features using `StandardScaler` (ML) or `MinMaxScaler` (ANN).  
  - Encoding of categorical features (binary, nominal, ordinal) inside a `ColumnTransformer`.  
  - All transformations are bundled into a **scikit‑learn `Pipeline`**, ensuring reproducibility and preventing data leakage.

- **Machine Learning Models (HDP-ML):**  
  - Benchmarking of 11 classifiers with consistent evaluation.  
  - Final model: a **Stacking Classifier** combining Random Forest, Gradient Boosting, AdaBoost, Extra Trees, and SVC, with a Logistic Regression meta‑learner.  
  - The entire preprocessing + model pipeline is saved as a single `.joblib` file for immediate deployment.

- **Artificial Neural Networks (HDP-ANN):**  
  - Multi‑layer feedforward architecture (16 → 16 → dropout 0.3 → 1 sigmoid).  
  - Trained with early stopping and model checkpointing.  
  - Custom decision threshold (0.45) to balance false negatives and false positives.  
  - Preprocessing pipeline saved separately to be combined with the Keras model during inference.

- **Evaluation Metrics:**  
  - Weighted F1‑score, accuracy, confusion matrix, and full classification reports for both approaches.

- **Organized Codebase:**  
  - Clear separation of concerns: preprocessing, training, and evaluation are modularised in `src/` and notebooks.  
  - Reusable utility functions and a shared custom transformer across both paradigms.

---

## How to Use / Run the Project

1. **Prepare the Dataset:**  
   Place the dataset CSV file in the `Data/` folder.

2. **Exploratory Data Analysis:**  
   Open `Notebooks/EDA.ipynb` to explore data distributions, outliers, and skewness.

3. **Run the ML Pipeline:**  
   - Open `Notebooks/classification/HDP-ML.ipynb`.  
   - Execute cells sequentially to perform row‑level cleaning, build the preprocessing pipeline, benchmark multiple models, train the final stacking ensemble, and evaluate it.  
   - The trained pipeline is saved automatically to `Models/classification/`.

4. **Run the ANN Pipeline:**  
   - Open `Notebooks/classification/HDP-ANN.ipynb`.  
   - Execute cells to apply the same cleaning, build the ANN‑specific pipeline (with `MinMaxScaler`), train the neural network, and evaluate.  
   - The fitted preprocessor and Keras model are saved for future use.

5. **Use the Source Code Directly:**  
   - Import the `preprocessor` and `trainer` classes from `Src/HDP-ML` or `Src/HDP-ANN` for programmatic use.  
   - Example usage is shown in the notebooks and can be adapted to scripts.

> **Note:** The project is designed to work with the provided dataset. All required dependencies are listed in `requirements.txt`.

---

## Conclusion and Results Comparison

In this project, we implemented two approaches to predict whether a patient has heart disease:

1. **Traditional Machine Learning Models (HDP-ML)**
   - Models used: Random Forest, Gradient Boosting, AdaBoost, Extra Trees, SVC, and a Stacking Classifier.
   - The final stacked model achieved a **weighted F1 score of ~89%** on the validation set.
   - These models performed very well despite the relatively small size of the dataset, showing strong generalization capabilities.

2. **Artificial Neural Network (HDP-ANN)**
   - Architecture: Multi-layer feedforward network with two hidden layers (16 units and 16 units) and dropout of 0.3 to prevent overfitting.
   - Training: 30 epochs with early stopping based on validation loss.
   - Evaluation: After converting predicted probabilities to binary labels using a 0.45 decision threshold, the ANN achieved a **weighted F1 score of ~87%**.
   - Confusion matrix analysis helped fine-tune the decision threshold, reducing false negatives while keeping false positives at an acceptable level.

---

### Key Observations:

- Both approaches yielded strong results, with only a small difference (~2–3%) in F1 scores.
- Traditional ML models slightly outperformed the neural network in this specific case, likely due to the small dataset size and effective feature engineering.
- The neural network provides flexibility for future expansion or integration of additional features or data, while traditional ML remains robust and efficient for smaller datasets.

> Overall, this project demonstrates the effectiveness of both traditional ML models and neural networks in predicting heart disease, highlighting preprocessing, feature engineering, and model evaluation strategies.


---

## References / Acknowledgments

- **Dataset**: The dataset used in this project is from [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It contains patient health metrics and indicators relevant to heart disease prediction.  
- **Libraries and Tools**: This project leverages Python libraries such as `pandas`, `numpy`, `scipy`, `scikit-learn`, `tensorflow`, `joblib`, `matplotlib`, and `seaborn` for data preprocessing, model training, evaluation, and visualization.  
- **Inspiration & Learning**: The project workflow and techniques were guided by standard practices in machine learning and deep learning, including feature scaling, encoding, stacking classifiers, neural network design, and evaluation strategies.  

