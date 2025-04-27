# Load libraries
library(dplyr)
library(readr)
library(psych)        # for describe
library(car)          # for F-test (leveneTest)
library(stats)

# Load dataset

# =========================================
# Basic Data Analysis for processed.csv
# =========================================

# 1. Load Libraries
library(tidyverse)  # For data manipulation and visualization
library(GGally)     # For quick plots
library(corrplot)   # For correlation plot

# 2. Load the Data
data <- read_csv("C:/Users/halim/Downloads/Haleeq/Project/Github/prediciton/test_1/data/processed.csv")

# 3. Quick Overview
str(data)           # Structure of the data
summary(data)       # Summary statistics
glimpse(data)       # Quick look

# 4. Check for Missing Values
colSums(is.na(data))  # Total NAs per column

# 5. Data Types Check
sapply(data, class)

# 6. Basic Exploratory Data Analysis (EDA)

# 6.1 Distribution of Numeric Variables
numeric_vars <- data %>% select(where(is.numeric))


# Histograms
numeric_vars %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = "Histograms of Numeric Features")

# 6.2 Correlation Matrix
correlation_matrix <- cor(numeric_vars)
corrplot(correlation_matrix, method = "color", type = "upper", tl.cex = 0.8)

# 6.3 Scatterplots: Key Features vs Particle Size

ggpairs(data, columns = c("extract_volume_mL_", "conc_M_", "precursor_volume_mL_", "pH", "time_hr_", "particle_size_nm_"))

# 7. Basic Analysis of Categorical Variables

# Count plots
categorical_vars <- data %>% select(where(is.character))

for (col in names(categorical_vars)) {
  p <- ggplot(data, aes_string(x = col)) +
    geom_bar(fill = "lightgreen", color = "black") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = paste("Distribution of", col))
  
  print(p)   # Print the plot object
}

# Boxplot: Particle Size vs Methods
# Filter only for 'co-precipitation' and 'stirring' methods
filtered_data <- data %>%
  filter(methods %in% c("Co-Precipitation", "Stirring"))

ggplot(filtered_data, aes(x = methods, y = particle_size_nm_)) +
  geom_boxplot(fill = "lightcoral", color = "black", outlier.color = "red") +
  theme_minimal() +
  labs(
    title = "Particle Size Distribution: Co-precipitation vs Stirring",
    x = "Synthesis Method",
    y = "Particle Size (nm)"
  ) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
  geom_jitter(width = 0.2, color = "blue", alpha = 0.5)  # Add jitter points to show spread


# Scatter Plot: Particle Size vs Methods
ggplot(filtered_data, aes(x = methods, y = particle_size_nm_)) +
  geom_jitter(width = 0.2, size = 2, color = "blue", alpha = 0.6) +  # scatter points
  theme_minimal() +
  labs(
    title = "Particle Size Scatter: Co-precipitation vs Stirring",
    x = "Synthesis Method",
    y = "Particle Size (nm)"
  ) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))


# 8. Save Cleaned Dataset (if needed)
# write.csv(data, "processed_clean.csv", row.names = FALSE)

