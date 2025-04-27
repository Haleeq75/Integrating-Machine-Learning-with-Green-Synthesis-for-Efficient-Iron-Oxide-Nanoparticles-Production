
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import os

# Step 2: Load Processed Data
df = pd.read_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/Main/data/processed.csv")

# Step 3: Separate Features and Target
target = 'particle_size_nm_'  # update if different
X = df.drop(columns=[target])
y = df[target]

# Step 4: Preprocessing (Encoding + Scaling)
cat_cols = ['plant_extract', 'precursor', 'methods', 'additives']
num_cols = [col for col in X.columns if col not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ("num", StandardScaler(), num_cols)
])

X_processed = preprocessor.fit_transform(X)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Regressor (RFR)
rfr = RandomForestRegressor(n_estimators=200, random_state=42)
rfr.fit(X_train, y_train)

# Step 7: Get RFR Predictions
rfr_train_preds = rfr.predict(X_train)
rfr_test_preds = rfr.predict(X_test)

# Save RFR predictions
os.makedirs("result", exist_ok=True)
pd.DataFrame({
    "Actual": y_test.values,
    "Predicted_RFR": rfr_test_preds
}).to_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/Main/results/rfr_predictions.csv", index=False)

print("\n Random Forest completed. Now feeding RFR output to ANN...")

# Step 8: Prepare data for ANN
# Here, input to ANN = RFR predictions
# Input for ANN: rfr_train_preds → predict y_train
# Validation for ANN: rfr_test_preds → predict y_test

# Reshape because ANN expects 2D input
rfr_train_preds = rfr_train_preds.reshape(-1, 1)
rfr_test_preds = rfr_test_preds.reshape(-1, 1)

# Step 9: Build and Train ANN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output Layer
])

model.compile(optimizer='adam', loss='mse')

model.fit(rfr_train_preds, y_train, epochs=200, batch_size=8, validation_data=(rfr_test_preds, y_test), verbose=0)

# Step 10: ANN Prediction
ann_preds = model.predict(rfr_test_preds).flatten()

# Save ANN predictions
pd.DataFrame({
    "Actual": y_test.values,
    "Predicted_ANN": ann_preds
}).to_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/Main/results/ann_predictions.csv", index=False)

print("\n ANN completed. All results saved into 'result' folder.")
