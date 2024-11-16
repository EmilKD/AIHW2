# Machine Learning Regression and Classification Models

This repository contains Python code implementations for regression and classification tasks on datasets related to car prices and diabetes prediction. The analysis explores data preprocessing, feature engineering, and various models for prediction, with detailed comparisons of their performance.

## Table of Contents
1. [Overview](#overview)
2. [Regression Analysis](#regression-analysis)
   - [Data Loading and Exploration](#data-loading-and-exploration)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Model Selection](#model-selection)
   - [Model Performance Evaluation](#model-performance-evaluation)
3. [Classification Analysis](#classification-analysis)
   - [Data Loading and Cleaning](#data-loading-and-cleaning)
   - [Data Visualization](#data-visualization)
   - [Model Training and Testing](#model-training-and-testing)
   - [Optimization Techniques](#optimization-techniques)
4. [Results and Comparisons](#results-and-comparisons)
5. [Conclusion](#conclusion)

---

## Overview
This project applies machine learning models for regression and classification tasks using the following datasets:
- **Car Price Dataset** for predicting car prices based on various attributes.
- **Diabetes Dataset** for predicting diabetes outcomes using patient health metrics.

### Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Regression Analysis

### Data Loading and Exploration
The dataset used for regression tasks is `CarPrice.csv`. Initial data exploration includes:
```python
data = pd.read_csv("CarPrice.csv", index_col=0, header=0)
print(data.info())
print(data.describe())
```
- Total entries: 205
- Features: 25 columns including numerical and categorical data.

### Data Preprocessing
- Categorical data transformation using one-hot encoding and indexing.
- Standardization and normalization of data using Scikit-learn's `StandardScaler`.

### Feature Engineering
Selected the top 10 features using `SelectKBest` from Scikit-learn:
```python
selector = SelectKBest(f_regression, k=10)
x_new = selector.fit_transform(x, y)
```

### Model Selection
Implemented the following regression models:
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Support Vector Regression (SVR)**

### Model Performance Evaluation
- Evaluation metrics: **R² Score** and **Root Mean Square Error (RMSE)**
- Models were compared on standardized data, with the results shown in the tables below.

| Model            | Training R² | Test R² |
|------------------|-------------|---------|
| Linear Regression| 0.8505      | 0.7812  |
| Lasso Regression | 0.8337      | 0.7727  |
| Ridge Regression | 0.8505      | 0.7812  |
| SVR              | 0.9266      | 0.7348  |

---

## Classification Analysis

### Data Loading and Cleaning
The dataset for classification is `diabetes.csv`. Initial data exploration includes counting and handling missing values:
```python
data = pd.read_csv("./diabetes.csv")
data.fillna(data.mean(), inplace=True)
```

### Data Visualization
Explored feature relationships using correlation matrices and joint plots:
```python
sns.heatmap(data.corr(), annot=False, fmt='.2f', cbar=True, cmap="Blues")
```

### Model Training and Testing
Implemented the following classification models:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**

### Optimization Techniques
Used GridSearchCV for hyperparameter tuning to improve model performance.

---

## Results and Comparisons
| Model                 | Initial Accuracy | Optimized Accuracy |
|-----------------------|------------------|--------------------|
| Logistic Regression   | 74.02%          | 78.17%             |
| K-Nearest Neighbors   | 66.88%          | 75.89%             |
| Decision Tree         | 62.33%          | 75.73%             |
| Random Forest         | 74.02%          | 91.2%              |
| SVM                   | 72.07%          | 76.87%             |

---

## Conclusion
- Regression models showed strong predictive capabilities for car prices, with linear and ridge regression achieving similar performance.
- Classification models were optimized for diabetes prediction, with Random Forest performing the best but also exhibiting overfitting tendencies.

For more details, please refer to the full code and accompanying visualizations in this repository.

---
