# Medical & Healthcare Prediction Framework

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-blue?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-success?logo=pandas)
![License](https://img.shields.io/github/license/Nimaabediforud/Heart-Disease-Prediction)
![Status](https://img.shields.io/badge/Status-Ongoing-success?style=flat-square)

---

## Overview

Medical-Healthcare-Prediction-Framework is a collection of machine learning and deep learning workflows developed for healthcare prediction tasks.

The repository contains exploratory data analysis, classification models, regression models, neural network implementations, and comparative evaluations across multiple medical datasets.

**The project emphasizes reproducibility, structured experimentation, and educational clarity.**

---

## Project Goals

This project was developed to explore and compare supervised learning techniques for healthcare prediction tasks.

The main objectives were:

- Build reproducible preprocessing pipelines for medical datasets.
- Compare traditional machine learning models and artificial neural networks.
- Investigate both classification and regression paradigms.
- Evaluate model performance using appropriate metrics and validation strategies.
- Develop a modular workflow suitable for future healthcare prediction projects.
---

## Datasets

### Heart Failure Prediction Dataset

Purpose:
- Binary classification
- Regression experiments

Target:
- HeartDisease (For classification)
- Cholesterol & Oldpeak (For regression)

Samples:
- 918

Source:
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

---

### Framingham Heart Study Dataset

Purpose:
- Exploratory analysis and preliminary regression experiments

Target:
- totChol (For regression)

Samples:
- 4240

Source:
https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset

---

### Medical Insurance Cost Dataset

Purpose:
- Regression

Target:
- Charges

Samples:
- 1338

Source:
https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset

---
## Project Structure
```
Medical-Healthcare-Prediction-Framework/
│
├── Data/
│
├── Models/
│   ├── Classification/
│   └── Regression/
│
├── Notebooks/
│   │
│   ├── EDA/
│   │   ├── EDA.ipynb
│   │   ├── Framingham-EDA.ipynb
│   │   └── Insurance-EDA.ipynb
│   │
│   ├── Classification/
│   │   ├── HDP-CLS-ML.ipynb
│   │   └── HDP-CLS-ANN.ipynb
│   │
│   ├── Regression/
│   │   ├── First-Reg/
│   │   │   └── REG-ML-HD.ipynb
│   │   │
│   │   ├── Second-Reg/
│   │   │   └── REG-ML-EXT1.ipynb
│   │   │
│   │   └── Third-Reg/
│   │       ├── REG-ML-EXT2.ipynb
│   │       └── REG-ANN-EXT2.ipynb
│   │
│   ├── Comparison-Results/
│   │   ├── Comparison-Results.ipynb
│   │   └── comparison_results.py
│   │
│   └── utils.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
---

## Current Status

### Completed

- Exploratory Data Analysis (EDA)
- Heart Disease Classification (Machine Learning)
- Heart Disease Classification (Artificial Neural Networks)
- Medical Cost Regression (Machine Learning)
- Medical Cost Regression (Artificial Neural Networks)
- Comparative Analysis Notebook
