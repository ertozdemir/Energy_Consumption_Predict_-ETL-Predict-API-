import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
accuracies = joblib.load(os.path.join(BASE_DIR, 'models', 'accuracies.pkl'))
linear_r2 = accuracies['Linear Regression r2']
linear_mse = accuracies['Linear Regression mse']
linear_mae = accuracies['Linear Regression mae']
random_forest_r2 = accuracies['Random Forest r2']
random_forest_mse = accuracies['Random Forest mse']
random_forest_mae = accuracies['Random Forest mae']
xgboost_r2 = accuracies['XGBoost r2']
xgboost_mse = accuracies['XGBoost mse']
xgboost_mae = accuracies['XGBoost mae']

print(f"Linear Regression: R2: {linear_r2}, MSE: {linear_mse}, MAE: {linear_mae}")
print(f"Random Forest: R2: {random_forest_r2}, MSE: {random_forest_mse}, MAE: {random_forest_mae}")
print(f"XGBoost: R2: {xgboost_r2}, MSE: {xgboost_mse}, MAE: {xgboost_mae}")