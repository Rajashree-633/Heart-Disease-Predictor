import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import joblib


#  LOAD THE DATA

data = pd.read_csv("HeartDiseaseTrain-Test.csv")
print("First 5 rows:")
print(data.head())


# HANDLE CATEGORICAL DATA

# Find all non-numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_cols)

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    print(f"Encoded '{col}'")


#  SPLIT FEATURES AND TARGET

# Assuming your target column is named 'target' or 'HeartDisease'
target_col = 'target' if 'target' in data.columns else 'HeartDisease'
X = data.drop(columns=[target_col])
y = data[target_col]


#  TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#  SCALE THE DATA

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  TRAIN THE MODEL

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


# EVALUATE MODEL

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# SAVE MODEL & SCALER

pickle.dump(model, open('heart_disease_predictor.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
joblib.dump(model, 'heart_disease_model.pkl')

print("\n Model and Scaler saved successfully!")