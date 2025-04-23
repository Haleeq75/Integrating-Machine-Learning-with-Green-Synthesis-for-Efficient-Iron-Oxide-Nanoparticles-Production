# Integrating Machine Learning with Green Synthesis for Efficient Iron Oxide Nanoparticles Production

This repository presents a machine learning-based framework to optimize the green synthesis of iron oxide nanoparticles using plant extracts. The goal is to predict and optimize synthesis parameters for achieving a desired particle size using Random Forest Regression and Artificial Neural Networks (ANN), followed by numerical optimization.

## Project Pipeline

1. **Import Libraries** – Load all necessary libraries for data handling, modeling, and optimization.
2. **Load & Clean Data** – Read the dataset and sanitize column names.
3. **Feature Engineering** – One-hot encode categorical features and scale numeric ones.
4. **Train ANN Model** – Build and train a neural network to predict particle size from synthesis parameters.
5. **Optimization** – Use SciPy's `minimize` to find the optimal parameters that lead to a target particle size.

## Dataset

- The dataset `green_synthesis_data.csv` consists of experimental records including:
  - Plant extract type
  - Precursor details
  - Additives
  - Synthesis methods
  - Numerical conditions (pH, temperature, time, etc.)
  - Target variable: Particle size in nanometers (`particle_size_nm`)

## Optimization Approach

Given a desired particle size, the trained ANN model is embedded in an optimization routine to reverse-engineer the input features (conditions) that would yield that particle size.

## Repository Structure

data/
    green_synthesis_data.csv
notebooks/
    main_pipeline.ipynb
src/
    model_pipeline.py
results/
    plots/
README.md
requirements.txt
LICENSE

## Dependencies

Install all requirements using:

pip install -r requirements.txt

## Notes

- Feature decoding for one-hot encoded inputs can be performed using:

    preprocessor.named_transformers_['cat'].categories_

- You can reverse-transform optimized parameters using a decoding function (to be added).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions

Feel free to open issues or contribute to the repository. Contributions related to green synthesis, ML model improvements, or additional optimization methods are welcome.
