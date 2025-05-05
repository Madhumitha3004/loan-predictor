# model_training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/train.csv')

# Drop rows with missing target
df.dropna(subset=['Loan_Status'], inplace=True)

# Map target
y = df['Loan_Status'].map({'Y': 1, 'N': 0})
X = df.drop(columns=['Loan_ID', 'Loan_Status'])

# Identify columns
numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric),
    ('cat', categorical_pipeline, categorical)
])

# Final pipeline with RandomForest
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Predict and evaluate
preds = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(pipeline, 'loan_model.pkl')

# Get feature importances
ohe = pipeline.named_steps['preprocess'].transformers_[1][1].named_steps['encoder']
cat_features = ohe.get_feature_names_out(categorical)
all_features = numeric + list(cat_features)
importances = pipeline.named_steps['model'].feature_importances_

# Save feature importance to CSV
fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
fi_df.sort_values(by='Importance', ascending=False, inplace=True)
fi_df.to_csv('feature_importance.csv', index=False)
