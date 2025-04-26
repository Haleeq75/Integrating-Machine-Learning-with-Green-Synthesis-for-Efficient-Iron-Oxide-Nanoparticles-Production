import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
import os

# Load processed data
df = pd.read_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/Main/data/Processed.csv")

# Define input and output
X = df[['particle_size_nm_']]  # particle size as input
y = df.drop(columns=['particle_size_nm_'])  # all other parameters as output


# Encode output parameters
cat_cols = ['plant_extract', 'precursor', 'methods', 'additives']
num_cols = [col for col in y.columns if col not in cat_cols]

preprocessor_y = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ("num", StandardScaler(), num_cols)
])

y_processed = preprocessor_y.fit_transform(y)

#Convert sparse matrix to dense array
y_processed = y_processed.toarray()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_processed, test_size=0.2, random_state=42)

# Train Random Forest for multi-output
rfr_inverse = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
rfr_inverse.fit(X_train, y_train)

# Predict parameters from desired particle size
particle_size= input("Enter expected particle size:")
expected_particle_size = [[30]]  # example: 30 nm
optimized_parameters = rfr_inverse.predict(expected_particle_size)

print("Predicted Parameters:", optimized_parameters)