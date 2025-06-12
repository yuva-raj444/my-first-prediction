import pandas as pd
import joblib

# Load trained model and label encoders
model = joblib.load("diabetes_model.pkl")
encoders = joblib.load("label_encoders.pkl")

le_gender = encoders['gender']
le_smoke = encoders['smoking_history']

# Load CSV (might include 'diabetes' column - remove it if present)
df = pd.read_csv("input_no_label.csv")
if 'diabetes' in df.columns:
    df = df.drop(columns=['diabetes'])

# Map smoking history values
smoke_map = {
    'never': 'never',
    'current': 'current',
    'former': 'former',
    'not current': 'former',
    'ever': 'former',
    'No Info': 'unknown'
}

# Apply and fix unknown values
df['smoking_history'] = df['smoking_history'].map(smoke_map)
df['smoking_history'] = df['smoking_history'].replace({'unknown': 'never'})

# Encode categorical features
df['gender'] = le_gender.transform(df['gender'])
df['smoking_history'] = le_smoke.transform(df['smoking_history'])

# Predict
predictions = model.predict(df)

# Add predictions to DataFrame
df['diabetes_prediction'] = predictions
df['diabetes_prediction_label'] = df['diabetes_prediction'].apply(lambda x: "Diabetic" if x == 1 else "Not Diabetic")

# Save results
df.to_csv("predicted_diabetes_output.csv", index=False)

# Preview
print("✅ Predictions saved to 'predicted_diabetes_output.csv'")
print(df[['age', 'gender', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes_prediction_label']].head(10))
