# predict_rfr_ann.py

# Step 1: Import Libraries
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd


# Step 2: Load Models
rfr_inverse = joblib.load("C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/results/rfr_inverse_model.pkl")
preprocessor_y = joblib.load("C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/results/preprocessor_y.pkl")
ann_model = load_model("C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/results/ann_model.h5", compile=False)

# Step 3: Define Decoding Function
def decode_parameters(predicted_array, preprocessor_y):
    cat_encoder = preprocessor_y.named_transformers_['cat']
    num_scaler = preprocessor_y.named_transformers_['num']
    
    n_cat = cat_encoder.transform([['dummy', 'dummy', 'dummy', 'dummy']]).shape[1]
    cat_pred = predicted_array[:, :n_cat]
    num_pred = predicted_array[:, n_cat:]
    
    cat_labels = cat_encoder.inverse_transform(cat_pred)
    num_labels = num_scaler.inverse_transform(num_pred)
    
    full_labels = np.concatenate([cat_labels, num_labels], axis=1)
    return full_labels

# Step 4: Predict for New Particle Size
particle_size=int(input("\nEnter desired particle size:"))
#desired_particle_size = [[particle_size]]  
desired_particle_size = pd.DataFrame([[particle_size]], columns=["particle_size_nm_"])

# Step 5: RFR Prediction (Intermediate Step)
rfr_pred = rfr_inverse.predict(desired_particle_size)

# Step 6: ANN Refinement
final_pred = ann_model.predict(rfr_pred)

# Step 7: Decode back to human readable
decoded_params = decode_parameters(final_pred, preprocessor_y)

# Step 8: Display
print("\nFinal Optimized Parameters for Particle Size", desired_particle_size.iloc[0, 0])
columns = ['plant_extract', 'precursor', 'methods', 'additives', 'volume_mL_', 'conc_M_', 'volume_mL_.1', 'ph', 'time_hr_']

for idx, col in enumerate(columns):
    print(f"{col}: {decoded_params[0, idx]}")
