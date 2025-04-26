# preprocessing.py

# Step 1: Import Required Libraries
import pandas as pd
from sklearn.impute import SimpleImputer

#C:\Users\halim\Downloads\Haleeq\Project\Github\Integrating-Machine-Learning-with-Green-Synthesis-for-Efficient-Iron-Oxide-Nanoparticles-Production\Main\data
# Step 2: Load Dataset
df = pd.read_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/Integrating-Machine-Learning-with-Green-Synthesis-for-Efficient-Iron-Oxide-Nanoparticles-Production/Main/data/raw.csv")

# Step 3: Clean Column Names
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df.columns = df.columns.str.strip()

# Step 4: Strip Whitespace from Object Columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Step 5: Identify Columns
target = 'particle_size_nm_'  # Update if needed
cat_cols = ['plant_extract', 'precursor', 'methods', 'additives']
num_cols = [col for col in df.columns if col not in cat_cols + [target]]

# Step 6: Handle Missing Values
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Step 7: Save Processed Data
df.to_csv("c:/Users/halim/Downloads/Haleeq/Project/Github/Integrating-Machine-Learning-with-Green-Synthesis-for-Efficient-Iron-Oxide-Nanoparticles-Production/Main/data/processed.csv", index=False)

print("✅ Preprocessing complete. Processed file saved to 'data/Processed.csv'")
