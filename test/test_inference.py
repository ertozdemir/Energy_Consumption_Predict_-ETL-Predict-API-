import joblib
import pandas as pd

# Mocking raw input data as it would come from a user or API
# Note: The model pipeline expects these column names
df_sample = pd.DataFrame({
    'building_type': ['Commercial'],
    'square_meters': [3300.0],  # Example numeric value
    'number_occupants': [5],
    'appliances_used': [10],   # Treated as categorical or string in DB but passthrough in model?
                                # In model.py, simple passthrough means it must be numeric-like if not one-hot encoded.
                                # Let's verify: In DB 'appliances_used' is String. 
                                # In model.py, we only one-hot encoded `building_type` and `day_of_week`.
                                # So `appliances_used` goes to 'passthrough'. 
                                # This might be an issue if it's a String like "5". 
                                # But Linear Regression/XGBoost needs numbers.
                                # If it's "5" (string), sklearn might error or XGBoost handles it?
                                # XGBoost handles it if enable_categorical=True, but Sklearn LinearRegression will FAIL on string.
                                # Let's see if model.py crashes first.
    'day_of_week': ['Weekend'] 
})

print("Loading linear_regression.pkl...")
try:
    model = joblib.load('linear_regression.pkl')
    print("Model loaded successfully.")
    
    print("Input Data:")
    print(df_sample)
    
    pred = model.predict(df_sample)
    print(f"\nPrediction (Energy Consumption): {pred[0]:.2f}")

except Exception as e:
    print(f"\nAN ERROR OCCURRED: {e}")
