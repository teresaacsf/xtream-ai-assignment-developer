from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

# Load models and preprocessing objects
model_linear = joblib.load('models/model_linear_model.joblib')
model_xgboost = joblib.load('models/model_xgboost_model.joblib')
scaler = joblib.load('models/scaler.joblib')
poly = joblib.load('models/poly.joblib')
feature_names = joblib.load('models/feature_names.joblib')
df_diamonds = pd.read_csv('data/diamonds.csv')

# Define the expected preprocessing steps
def preprocess_input(data, scaler, feature_names, poly=None):
    df = pd.DataFrame([data])
    
    # Drop irrelevant columns if they are present
    df = df.drop(columns=['depth', 'table', 'y', 'z'], errors='ignore')
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
    
    # Add missing columns with zeros
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_names]
    
    # Scale the data using the saved StandardScaler
    df_scaled = scaler.transform(df)
    
    # Apply polynomial features if provided
    if poly:
        df_poly = poly.transform(df_scaled)
        return df_scaled, df_poly
    
    return df_scaled, df_scaled

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(Exception)
def handle_error(e):
    response = {'error': str(e)}
    return jsonify(response), 500

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()

        # Validate input data
        required_fields = ['carat', 'cut', 'color', 'clarity']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: '{field}'")

        # Preprocess input data
        X_pred_scaled, X_pred_poly = preprocess_input(data, scaler, feature_names, poly)

        # Log the preprocessed data
        logging.debug(f"Preprocessed input data for linear model: {X_pred_poly}")
        logging.debug(f"Preprocessed input data for XGBoost model: {X_pred_scaled}")

        # Predict with linear model
        pred_linear_log = model_linear.predict(X_pred_poly)
        logging.debug(f"Linear model log prediction: {pred_linear_log}")

        # Check for exp overflow/underflow
        pred_linear = np.exp(pred_linear_log) if pred_linear_log < 700 else float('inf')
        logging.debug(f"Linear model prediction after exp: {pred_linear}")

        # Predict with XGBoost model
        pred_xgboost_log = model_xgboost.predict(X_pred_scaled)
        pred_xgboost = np.exp(pred_xgboost_log)
        logging.debug(f"XGBoost model log prediction: {pred_xgboost_log}")
        logging.debug(f"XGBoost model prediction: {pred_xgboost}")

        # Log the coefficients of the linear regression model
        logging.debug(f"Linear model coefficients: {model_linear.coef_}")
        logging.debug(f"Linear model intercept: {model_linear.intercept_}")

        # Prepare response
        response = {
            'linear_regression_prediction': float(pred_linear[0]),
            'xgboost_prediction': float(pred_xgboost[0])
        }

        return jsonify(response)

    except ValueError as ve:
        logging.exception("ValueError in prediction")
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        logging.exception("Unexpected error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/similar_samples', methods=['POST'])
def get_similar_samples():
    try:
        data = request.get_json()
        required_fields = ['cut', 'color', 'clarity', 'carat', 'n_samples']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: '{field}'")

        cut = data['cut']
        color = data['color']
        clarity = data['clarity']
        carat = float(data['carat'])  # Ensure carat is treated as a float
        n_samples = int(data['n_samples'])  # Ensure n_samples is treated as an int

        similar_df = df_diamonds[
            (df_diamonds['cut'] == cut) &
            (df_diamonds['color'] == color) &
            (df_diamonds['clarity'] == clarity)
        ]

        closest_weight = similar_df.iloc[(similar_df['carat'] - carat).abs().argsort()[:n_samples]]
        similar_samples = closest_weight.to_dict(orient='records')

        return jsonify(similar_samples)

    except ValueError as ve:
        logging.exception("ValueError in similar samples")
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        logging.exception("Unexpected error")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
