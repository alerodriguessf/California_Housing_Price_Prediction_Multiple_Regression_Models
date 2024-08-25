
# California Housing Price Prediction with Multiple Regression Models

## Introduction

This project focuses on predicting California housing prices using three different regression models: Linear Regression, Support Vector Regression (SVR), and XGBoost Regressor. The California Housing dataset is used for this purpose, and the performance of each model is evaluated using common metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). Additionally, hyperparameter tuning is performed on the XGBoost model using GridSearchCV to optimize its performance.

## Code Breakdown

### 1. **Importing Necessary Libraries**
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
```
- **Purpose**: This section imports the essential Python libraries required for data handling, model building, evaluation, and optimization.
  - **Pandas** and **NumPy** are used for data manipulation and numerical operations.
  - **Scikit-learn** provides tools for fetching the dataset, splitting the data, building models, and evaluating their performance.
  - **XGBoost** is used for implementing the XGBoost Regressor model.
  - **GridSearchCV** helps in hyperparameter tuning.

### 2. **Loading the California Housing Dataset**
```python
housing = fetch_california_housing()
```
- **Purpose**: The `fetch_california_housing()` function loads the California Housing dataset, which includes features and target values related to housing prices in California.

### 3. **Exploring the Dataset**
```python
print(housing)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df.head()
housing.keys()
print(housing.DESCR)
print(housing.target)
```
- **Purpose**: This block performs initial exploration:
  - Converts the dataset to a pandas DataFrame for easier manipulation.
  - Prints the first few rows to inspect the data.
  - Displays the dataset's keys and a description to understand its features and target variable.

### 4. **Defining Features and Target Variable**
```python
x = housing.data
y = housing.target
```
- **Purpose**: The dataset is split into features (`x`) and the target variable (`y`), where `x` includes independent variables used to predict `y`, which represents median house prices.

### 5. **Splitting the Dataset into Training and Testing Sets**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
- **Purpose**: This step splits the data into:
  - **Training Set**: 80% of the data used to train the model.
  - **Testing Set**: 20% of the data reserved for evaluating the model's performance.

### 6. **Implementing the Linear Regression Model**
```python
regLinear = LinearRegression().fit(x_train, y_train)
```
- **Purpose**: A Linear Regression model is trained on the training data to establish a linear relationship between the features and the target variable.

### 7. **Evaluating the Linear Regression Model**
```python
ylinear = regLinear.predict(x_test)
MSElinear = mean_squared_error(y_test, ylinear)
print("MSE Linear:", MSElinear)
print("RMSE Linear:", np.sqrt(MSElinear))
```
- **Purpose**: The trained Linear Regression model is used to predict house prices on the test set. The model’s performance is evaluated using MSE and RMSE.

### 8. **Implementing the Support Vector Regression (SVR) Model**
```python
regsvr = SVR().fit(x_train, y_train)
```
- **Purpose**: A Support Vector Regression (SVR) model is trained on the training data to capture more complex, non-linear relationships.

### 9. **Evaluating the SVR Model**
```python
ysvr = regsvr.predict(x_test)
msesvr = mean_squared_error(y_test, ysvr)
print("MSE SVR:", msesvr)
print("RMSE SVR:", np.sqrt(msesvr))
```
- **Purpose**: The SVR model is used to predict prices on the test set, and its performance is evaluated using MSE and RMSE.

### 10. **Implementing the XGBoost Regressor**
```python
regxgb = XGBRegressor().fit(x_train, y_train)
```
- **Purpose**: The XGBoost Regressor, a highly efficient implementation of gradient boosting, is trained on the training data. This model is known for its high performance, especially with large datasets.

### 11. **Evaluating the XGBoost Model**
```python
yxgb = regxgb.predict(x_test)
msexgb = mean_squared_error(y_test, yxgb)
print("MSE XGB:", msexgb)
print("RMSE XGB:", np.sqrt(msexgb))
```
- **Purpose**: Predictions are made using the XGBoost model on the test set, and its performance is evaluated using MSE and RMSE.

### 12. **Hyperparameter Tuning using GridSearchCV**
```python
parameters = {
    "max_depth": [5, 6, 7],
    "learning_rate": [0.1, 0.2, 0.3],
    "objective": ['reg:squarederror'],
    "booster": ['gbtree'],
    "n_jobs": [5],
    "gamma": [0, 1],
    "min_child_weight": [1, 3],
    "max_delta_step": [0, 1],
    "subsample": [0.5, 1],
}
xgbGrid = GridSearchCV(XGBRegressor(), parameters, refit="neg_mean_squared_error", verbose=True)
xgbgridmodel = xgbGrid.fit(x_train, y_train)
```
- **Purpose**: GridSearchCV is used to perform an exhaustive search over a specified parameter grid for the XGBoost model, aiming to find the optimal combination of hyperparameters to minimize MSE.

### 13. **Evaluating the Optimized XGBoost Model**
```python
ygrid = xgbgridmodel.predict(x_test)
msegrid = mean_squared_error(y_test, ygrid)
print("MSE Grid:", msegrid)
print("RMSE Grid:", np.sqrt(msegrid))
```
- **Purpose**: The optimized XGBoost model, tuned by GridSearchCV, is evaluated using the test set. The model's performance is assessed using MSE and RMSE to determine the effectiveness of the hyperparameter tuning.

---

This README now includes both a high-level overview and a detailed explanation of the code, making it comprehensive and accessible to anyone reviewing or using the code.
