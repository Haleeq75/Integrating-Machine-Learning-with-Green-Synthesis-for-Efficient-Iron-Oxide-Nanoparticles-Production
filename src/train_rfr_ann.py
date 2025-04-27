# train_rfr_ann.py

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Step 2: Load Processed Data
df = pd.read_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/data/processed.csv")

# Step 3: Define Inputs and Outputs
X = df[['particle_size_nm_']]  # Input: Particle size
y = df.drop(columns=['particle_size_nm_'])  # Output: Synthesis Parameters

# Step 4: Preprocessing y
cat_cols = ['plant_extract', 'precursor', 'methods', 'additives']
num_cols = [col for col in y.columns if col not in cat_cols]

preprocessor_y = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# Apply Preprocessing
y_processed = preprocessor_y.fit_transform(y)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_processed, test_size=0.2, random_state=42)

# Step 6: Train RFR Inverse Model
rfr_inverse = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
rfr_inverse.fit(X_train, y_train)

# Step 7: Use RFR output as ANN input
rfr_output_train = rfr_inverse.predict(X_train)
rfr_output_test = rfr_inverse.predict(X_test)

# Step 8: Build ANN Model
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(rfr_output_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_processed.shape[1], activation='linear')
])

ann_model.compile(optimizer='adam', loss='mse')

# Step 9: Train ANN
ann_model.fit(rfr_output_train, y_train, epochs=100, batch_size=16, validation_data=(rfr_output_test, y_test))

# Step 10: Save Models
joblib.dump(rfr_inverse, "C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/results/rfr_inverse_model.pkl")
joblib.dump(preprocessor_y, "C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/results/preprocessor_y.pkl")
ann_model.save("C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/results/ann_model.h5")

print("\n RFR+ANN models trained and saved successfully!") 
