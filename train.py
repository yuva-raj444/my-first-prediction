import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==== 1. Load the Dataset ====
df = pd.read_csv("diabetes_prediction_dataset.csv")  # Ensure this file exists in the same folder

# ==== 2. Clean 'smoking_history' Values ====
smoke_map = {
    'never': 'never',
    'current': 'current',
    'former': 'former',
    'not current': 'former',
    'ever': 'former',
    'No Info': 'unknown'
}
df['smoking_history'] = df['smoking_history'].map(smoke_map)

# ==== 3. Handle Missing Values ====
df.fillna(df.mean(numeric_only=True), inplace=True)

# ==== 4. Select Features and Target Manually ====
features = [
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]
target = 'diabetes'

X = df[features]
y = df[target]

# ==== 5. Encode Categorical Features ====
label_encoders = {}

# Encode 'gender'
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'])
label_encoders['gender'] = le_gender

# Encode 'smoking_history'
le_smoke = LabelEncoder()
X['smoking_history'] = le_smoke.fit_transform(X['smoking_history'])
label_encoders['smoking_history'] = le_smoke

# ==== 6. Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 7. Train the Model ====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==== 8. Evaluate Model ====
y_pred = model.predict(X_test)
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ==== 9. Save the Model ====
joblib.dump(model, 'diabetes_model.pkl')
print("\n📁 Trained model saved as: diabetes_model.pkl")

# ==== 10. Save Label Encoders ====
joblib.dump(label_encoders, 'label_encoders.pkl')
print("📁 Label encoders saved as: label_encoders.pkl")
