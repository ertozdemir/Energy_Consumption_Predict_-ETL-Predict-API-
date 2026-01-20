import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

def load_df():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database', 'energy_consumption.db'))
    query = "SELECT * FROM energy_data"
    df = pd.read_sql_query(query, conn)
    print('load is success')
    return df

def model_train(df):
    
    X = df.drop(['id', 'energy_consumption'], axis=1)
    y = df['energy_consumption']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =============================================================================
    # REFACTORED SECTION START
    # =============================================================================
    
    # 1. Preprocessing Pipeline
    # Using OneHotEncoder instead of LabelEncoder for categorical features.
    # LabelEncoder is typically for targets, not features, and imposes arbitrary order.
    # OneHotEncoder is better for Linear Regression and often XGBoost/RF.
    categorical_features = ['building_type', 'day_of_week'] 
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    

    # --- Linear Regression ---
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    print("Training Linear Regression...")
    lr_pipeline.fit(X_train, y_train)
    prediction_linear = lr_pipeline.predict(X_test)

    # --- Random Forest ---
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    rf_param_grid = {
        'regressor__n_estimators': [100, 200, 350, 500],
        'regressor__max_depth': [10, 20, 30, 50],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__bootstrap': [True, False]
    }

    print("Tuning Random Forest...")
    rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=10, cv=5, n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    best_rf_pipeline = rf_search.best_estimator_
    prediction_rf = best_rf_pipeline.predict(X_test)

    # --- XGBoost ---
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor())
    ])

    
    xgb_param_grid = {
        'regressor__n_estimators': [100, 200, 350, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 6, 10],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0]
    }

    print("Tuning XGBoost...")
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_param_grid, n_iter=10, cv=5, n_jobs=-1, verbose=1)
    xgb_search.fit(X_train, y_train)
    best_xgb_pipeline = xgb_search.best_estimator_
    prediction_xgb = best_xgb_pipeline.predict(X_test)

    # --- Metrics ---
    def print_metrics(name, y_true, y_pred):
        print(f'\n{name} Metrics')
        print('R2 Score:', r2_score(y_true, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_true, y_pred)) 
        print('Mean Absolute Error:', mean_absolute_error(y_true, y_pred))

    accuracy_list = {
        'Linear Regression r2': r2_score(y_test, prediction_linear),
        'Random Forest r2': r2_score(y_test, prediction_rf),
        'XGBoost r2': r2_score(y_test, prediction_xgb),
        'Linear Regression mse': mean_squared_error(y_test, prediction_linear),
        'Random Forest mse': mean_squared_error(y_test, prediction_rf),
        'XGBoost mse': mean_squared_error(y_test, prediction_xgb),
        'Linear Regression mae': mean_absolute_error(y_test, prediction_linear),
        'Random Forest mae': mean_absolute_error(y_test, prediction_rf),
        'XGBoost mae': mean_absolute_error(y_test, prediction_xgb)
    }

    print_metrics('Linear Regression', y_test, prediction_linear)
    print_metrics('Random Forest', y_test, prediction_rf)
    print_metrics('XGBoost', y_test, prediction_xgb)

    # --- Save Models ---
    # Saving the full pipelines. In predict.py, simple load and call predict(df).
    joblib.dump(lr_pipeline, os.path.join(BASE_DIR, 'models', 'linear_regression.pkl'))
    joblib.dump(best_rf_pipeline, os.path.join(BASE_DIR, 'models', 'random_forest.pkl'))
    joblib.dump(best_xgb_pipeline, os.path.join(BASE_DIR, 'models', 'xgboost.pkl'))
    joblib.dump(accuracy_list, os.path.join(BASE_DIR, 'models', 'accuracies.pkl'))
    print("\nModels and metrics saved as .pkl files (Pipeline objects).")


if __name__ == '__main__':
    df = load_df()
    model_train(df)


# =============================================================================
# USER'S ORIGINAL CODE (ARCHIVED FOR REFERENCE)
# =============================================================================
"""
def model_train_original(df):
    X = df.drop(['id', 'energy_consumption'], axis=1)
    y = df['energy_consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #encoding 
    le = LabelEncoder()
    transformer = ColumnTransformer(transformers=[
            ('encoder', le, ['building_type', 'day_of_week'])
        ], 
        remainder='passthrough'
    )

    X_train_encoded = transformer.fit_transform(X_train)
    X_test_encoded = transformer.transform(X_test)
    
    #Linear Regression 
    linear = LinearRegression()
    linear.fit(X_train_encoded, y_train)
    prediction_linear = linear.predict(X_test_encoded)
    
    #Random Forest Regressor
    rf = RandomForestRegressor()
# Parameter Tuning for Random Forest Regressor
    param_grid = {
        'n_estimators': [100, 200, 350, 500],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                             n_iter=10, cv=5, n_jobs=-1) 
    random_search.fit(X_train_encoded, y_train)
    best_rf = random_search.best_estimator_
    prediction_rf = best_rf.predict(X_test_encoded)

# XGBoost Regressor
    xgb = XGBRegressor()
    xgb.fit(X_train_encoded, y_train)
    prediction_xgb = xgb.predict(X_test_encoded)

# Parameter Tuning for XGBoost Regressor
    param_grid = {
        'n_estimators': [100, 200, 350, 500],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid,
                             n_iter=10, cv=5, n_jobs=-1) 
    random_search.fit(X_train_encoded, y_train)
    best_xgb = random_search.best_estimator_
    prediction_xgb = best_xgb.predict(X_test_encoded)

# metrics
    print('Linear Regression Metrics')
    print('R2 Score:', r2_score(y_test, prediction_linear))
    print('Mean Squared Error:', mean_squared_error(y_test, prediction_linear))
    print('Mean Absolute Error:', mean_absolute_error(y_test, prediction_linear))
    print('\\n')
    print('Random Forest Regressor Metrics')
    print('R2 Score:', r2_score(y_test, prediction_rf))
    print('Mean Squared Error:', mean_squared_error(y_test, prediction_rf))
    print('Mean Absolute Error:', mean_absolute_error(y_test, prediction_rf))
    print('\\n')
    print('XGBoost Regressor Metrics')
    print('R2 Score:', r2_score(y_test, prediction_xgb))
    print('Mean Squared Error:', mean_squared_error(y_test, prediction_xgb))
    print('Mean Absolute Error:', mean_absolute_error(y_test, prediction_xgb))

# Save models
    joblib.dump(linear, 'linear_regression.pkl')
    joblib.dump(best_rf, 'random_forest.pkl')
    joblib.dump(best_xgb, 'xgboost.pkl')
    
    return prediction_linear, prediction_rf, prediction_xgb
"""