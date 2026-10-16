# House Price Prediction Using Machine Learning

## 1. Project Overview

This project predicts the selling price of a house using machine learning. It uses the Ames Housing Dataset, which contains information about houses, such as their quality, living area, year built, garage, neighborhood, and other features.

Two machine learning models are used:
- Linear Regression
- Random Forest Regressor

The models are evaluated and compared to identify which one performs better.

## 2. Project Objectives

- Understand and explore housing data.
- Handle missing values and categorical features.
- Train machine learning regression models.
- Evaluate model performance using MAE, RMSE, and R².
- Find the minimum and maximum house prices.
- Predict house prices using house features.

## 3. Dataset

**Dataset:** Ames Housing Dataset

**Source:** [Download from Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

The original dataset contains 2,930 rows and 82 columns.

- **Input features:** House characteristics.
- **Target column:** `SalePrice`, the actual selling price of a house.
- **Removed columns:** `Order` and `PID`, which are identifiers rather than useful house characteristics for this project.
- **Features used:** 79 input columns before encoding categorical variables.

## 4. Tools and Libraries

- Python
- Google Colab
- Pandas — loading and exploring data
- NumPy — numerical calculations
- Matplotlib and Seaborn — data visualization
- Scikit-learn — preprocessing, training, and evaluation
- Joblib — saving and loading trained models

## 5. How to Implement the Project

### Step 1: Open Google Colab

Visit [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Install or import the required libraries

Scikit-learn and the other listed libraries are generally available in Google Colab.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

### Step 3: Upload and load the dataset

Download the CSV file from Kaggle, then upload `AmesHousing.csv` to your Colab session.

```python
from google.colab import files
uploaded = files.upload()
```

Select the dataset file and load it:

```python
df = pd.read_csv('/content/AmesHousing.csv')

print(df.head())
print("Dataset shape:", df.shape)
print(df.isnull().sum().sort_values(ascending=False).head(15))
```

### Step 4: Prepare the input features and target

Remove the identifier columns and separate the house features from the target price.

```python
df = df.drop(columns=['Order', 'PID'])

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
```

### Step 5: Split the dataset

Use 80% of the data for training and 20% for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
```

### Step 6: Preprocess the data

- Fill missing numerical values with the median.
- Fill missing categorical values with the most frequent value.
- Convert categorical features into numerical features using OneHotEncoder.

```python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### Step 7: Train Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
```

### Step 8: Evaluate Linear Regression

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

### Step 9: Train Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)
rf_pred = rf_model.predict(X_test_processed)
```

### Step 10: Evaluate Random Forest

```python
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2)
```

### Step 11: Compare the models

Use a comparison table to review both models' evaluation results. Lower MAE and RMSE are better, while a higher R² is generally better.

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 17,553.00 | 15,731.10 |
| RMSE | 30,216.15 | 26,335.19 |
| R² | 0.8861 | 0.9135 |

In this experiment, Random Forest performed better on the test set, with an R² score of 0.9135 (91.35%).

Actual results may vary if the dataset, preprocessing, or model settings change.

### Step 12: Find the house-price range

```python
min_price = df['SalePrice'].min()
max_price = df['SalePrice'].max()
avg_price = df['SalePrice'].mean()

print(f"Minimum House Price: ${min_price:,.0f}")
print(f"Maximum House Price: ${max_price:,.0f}")
print(f"Average House Price: ${avg_price:,.2f}")
```

In the tested dataset:
- Minimum price: $12,789
- Maximum price: $755,000

### Step 13: Test a house-price prediction

Use a house from the test dataset to check the prediction.

```python
sample_house = X_test.iloc[[0]].copy()

processed_house = preprocessor.transform(sample_house)
predicted_price = rf_model.predict(processed_house)[0]

print(f"Predicted House Price: ${predicted_price:,.2f}")
print(f"Actual House Price: ${y_test.iloc[0]:,.2f}")
```

To test changes to selected features, copy a sample house and update values such as overall quality, living area, garage capacity, and year built. The remaining features must also be supplied to the model.

### Step 14: Save the trained model and preprocessing pipeline

```python
import joblib

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(rf_model, 'random_forest_house_price_model.pkl')
joblib.dump(preprocessor, 'house_price_preprocessor.pkl')
```

These files allow the trained models and preprocessing pipeline to be loaded later without retraining, provided they are used with compatible data and library versions.

## 6. Evaluation Metrics Explained

- **MAE:** Average absolute difference between actual and predicted prices.
- **RMSE:** Measures prediction error while penalizing larger errors more heavily.
- **R² Score:** Measures how well the model explains variation in house prices.

A confusion matrix is not appropriate for this regression task because the model predicts continuous prices rather than classes.

## Prediction Error Analysis

I performed error analysis on the test-set predictions from both Linear Regression and Random Forest.

### Model Performance

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | $17,553.00 | $15,731.10 |
| RMSE | $30,216.15 | $26,335.19 |
| R² Score | 0.89 | 0.91 |

Random Forest performed better overall, with lower MAE and RMSE and a higher R² score.

### Largest Prediction Errors

I identified the 10 houses with the largest absolute prediction errors for each model. House 2181 had the largest error in both models.

- Actual price: $184,750
- Linear Regression prediction: $540,656.49
- Linear Regression absolute error: $355,906.49
- Random Forest prediction: $436,173.81
- Random Forest absolute error: $251,423.81

### Possible Reasons for Errors

The models may struggle with unusual combinations of property features, limited comparable examples, or sale conditions that are difficult to capture from the available input features. These are possible explanations and require further investigation.

### Conclusion

Random Forest performed better than Linear Regression on the test set, but both models made substantial errors for some properties. Error analysis helps identify these cases and areas for future model improvement.

## 7. Project Workflow

1. Load and explore the dataset.
2. Separate input features and target price.
3. Split the data into training and testing sets.
4. Handle missing values and encode categorical features.
5. Train Linear Regression.
6. Train Random Forest Regressor.
7. Evaluate and compare both models.
8. Display the price range.
9. Test predictions using house features.
10. Save the trained models and preprocessing pipeline.

## 8. Conclusion

This project demonstrates how machine learning can estimate house prices from housing characteristics. Linear Regression and Random Forest Regressor were compared using MAE, RMSE, and R². In the recorded experiment, Random Forest achieved the better evaluation results.

This is an educational prediction project. Predictions are estimates and should not be treated as guaranteed market prices.

## 9. Author

Created as a beginner-friendly machine learning project using Python and Google Colab.
