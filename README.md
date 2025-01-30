# Wine Quality Classification

## Overview

This project implements a machine learning pipeline to classify wine quality using various classification algorithms, including Logistic Regression, Random Forest, SVM, and XGBoost. The dataset used is the Wine Quality dataset from the UCI Machine Learning Repository.

## Dataset

The dataset consists of 1599 samples of red wine with 11 physicochemical features and one target variable (quality). The target variable is binarized into two classes:

- `1` (good quality): if quality >= 7
- `0` (bad quality): if quality < 7

The dataset is loaded from:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

## Dependencies

Ensure you have the following libraries installed before running the code:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Steps in the Pipeline

### 1. Import Libraries

The necessary libraries for data manipulation, visualization, and machine learning models are imported.

### 2. Load Dataset

The dataset is loaded and inspected using `pandas` to understand its structure and statistical properties.

### 3. Exploratory Data Analysis (EDA)

- Display dataset structure
- Generate summary statistics
- Visualize data distribution (e.g., histogram of alcohol content, target class distribution)
- Compute correlation matrix

### 4. Data Preprocessing

- Separate features (X) and target variable (y)
- Binarize the target variable (`quality`)
- Split data into training and testing sets (70%-30%)
- Standardize the features using `StandardScaler`

### 5. Model Training and Hyperparameter Tuning

Four machine learning models are trained and optimized using `GridSearchCV`:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**

Each model undergoes hyperparameter tuning using 5-fold cross-validation.

### 6. Model Evaluation

For each model, the following metrics are computed:

- **Accuracy**
- **ROC-AUC Score**
- **Confusion Matrix**
- **Classification Report**

### 7. Results Visualization

The following visualizations are generated:

1. **Comparison of Accuracy and ROC-AUC scores** (bar chart)
2. **ROC Curve for all models**
3. **Confusion Matrices for each model**
4. **Feature Importance for Random Forest and XGBoost**

## Running the Code

To execute the script, simply run:

```bash
python wine_quality_classification.py
```

## Expected Output

The script will output evaluation metrics and plots, showing model performances and feature importances.

## Author

This project was developed as part of a machine learning experiment in wine quality classification.

## License

This project is open-source and free to use for educational and research purposes.

