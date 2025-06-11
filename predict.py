import joblib
import pandas as pd

# ==== 1. Load model and encoders ====
model = joblib.load('diabetes_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# ==== 2. Helper function to handle yes/no inputs ====
def get_yes_no_input(prompt):
    while True:
        val = input(prompt + " (yes/no): ").strip().lower()
        if val in ['yes', 'no']:
            return 1 if val == 'yes' else 0
        else:
            print("❌ Please enter 'yes' or 'no'.")

# ==== 3. Get user input ====
gender = input("Enter gender (Male/Female): ").strip().capitalize()
age = float(input("Enter age: "))
hypertension = get_yes_no_input("Do you have hypertension?")
heart_disease = get_yes_no_input("Do you have any heart disease?")
smoking_history = input("Smoking history (never/current/former/not current/ever/No Info): ").strip().lower()
bmi = float(input("Enter BMI: "))
hba1c = float(input("Enter HbA1c level: "))
glucose = float(input("Enter blood glucose level: "))

# ==== 4. Clean smoking history ====
smoke_map = {
    'never': 'never',
    'current': 'current',
    'former': 'former',
    'not current': 'former',
    'ever': 'former',
    'no info': 'unknown'
}
smoking_history_clean = smoke_map.get(smoking_history, 'unknown')

# ==== 5. Encode categorical features ====
gender_encoded = encoders['gender'].transform([gender])[0]
smoke_encoded = encoders['smoking_history'].transform([smoking_history_clean])[0]

# ==== 6. Prepare input DataFrame ====
X_input = pd.DataFrame([{
    'gender': gender_encoded,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': smoke_encoded,
    'bmi': bmi,
    'HbA1c_level': hba1c,
    'blood_glucose_level': glucose
}])

# ==== 7. Predict ====
prediction = model.predict(X_input)[0]

# ==== 8. Output ====
print("\n🩺 Prediction Result:")
print(f"→ {'Diabetic' if prediction == 1 else 'Not Diabetic'} (value: {prediction})")
