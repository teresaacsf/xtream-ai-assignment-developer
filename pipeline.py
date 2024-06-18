

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import xgboost as xgb
import optuna
import json
from datetime import datetime

# Paths for data, models, and logs
DATA_PATH = 'data/diamonds.csv'
MODEL_PATH = 'models/'
LOG_PATH = 'logs/model_history.json'

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Function to fetch fresh data
def fetch_fresh_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Function to preprocess data
def preprocess_data(df):
    df = df.drop(columns=['depth', 'table', 'y', 'z'])
    df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
    df = df[df['price'] > 0]
    X = df.drop(columns=['price'])
    y = df['price']
    return X, y

# Function to train linear regression model with polynomial features
def train_linear_model(X, y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = Ridge(alpha=1.0)  # Using Ridge regression for regularization
    model.fit(X_poly, np.log(y))
    return model, poly

# Function to train XGBoost model with hyperparameter optimization
def train_xgboost_model(X_train, y_train):
    def objective(trial):
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, np.log(y_train))
        preds = model.predict(X_train)
        mae = mean_absolute_error(np.exp(np.log(y_train)), np.exp(preds))
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, np.log(y_train))
    return best_model

# Function to evaluate model
def evaluate_model(model, X, y, poly=None):
    if poly:
        X = poly.transform(X)
    y_pred_log = model.predict(X)
    y_pred_log = np.clip(y_pred_log, -100, 100)  # Clip predictions to a reasonable range
    y_pred = np.exp(y_pred_log)  # Transform predictions back to the original scale
    y_pred = np.nan_to_num(y_pred, nan=np.nanmedian(y_pred), posinf=np.nanmax(y_pred), neginf=np.nanmin(y_pred))  # Handle any remaining infinities
    y_pred = np.clip(y_pred, 0, 1e10)  # Additional clipping to avoid extremely large values
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return mse, r2, mae

# Function to save model and logs
def save_model_and_logs(model, scaler, poly, feature_names, metrics, model_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{MODEL_PATH}model_{model_name}.joblib"
    log_entry = {
        'model_name': model_name,
        'timestamp': timestamp,
        'mse': metrics[0],
        'r2': metrics[1],
        'mae': metrics[2]
    }
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.joblib'))
    joblib.dump(poly, os.path.join(MODEL_PATH, 'poly.joblib'))
    joblib.dump(feature_names, os.path.join(MODEL_PATH, 'feature_names.joblib'))
    
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w') as f:
            json.dump([], f)
    
    with open(LOG_PATH, 'r+') as f:
        log = json.load(f)
        log.append(log_entry)
        f.seek(0)
        json.dump(log, f, indent=4)

# Function to run the pipeline and select the best model
def pipeline(num_trials=10):
    df = fetch_fresh_data()
    X, y = preprocess_data(df)
    
    best_model = None
    best_mae = float('inf')
    best_metrics = None
    best_model_name = None
    best_poly = None

    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        linear_model, poly = train_linear_model(X_train_scaled, y_train)
        linear_metrics = evaluate_model(linear_model, X_test_scaled, y_test, poly)
        
        # xgb_model = train_xgboost_model(X_train, y_train)
        # xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        
        if linear_metrics[2] < best_mae:
            best_mae = linear_metrics[2]
            best_model = linear_model
            best_metrics = linear_metrics
            best_model_name = 'linear_model'
        
        # if xgb_metrics[2] < best_mae:
        #     best_mae = xgb_metrics[2]
        #     best_model = xgb_model
        #     best_metrics = xgb_metrics
        #     best_model_name = 'xgboost_model'
    save_model_and_logs(best_model, scaler, poly, X.columns.tolist(), best_metrics, best_model_name)

pipeline()
