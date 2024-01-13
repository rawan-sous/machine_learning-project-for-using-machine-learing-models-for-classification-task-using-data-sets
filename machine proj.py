import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
data = pd.read_csv("creditcard.csv")  # Replace with your dataset's filename

# Separate features from target variable
X = data.drop("Class", axis=1)  # Define X before using it

# Feature reduction using SelectKBest with f_classif
selector = SelectKBest(f_classif, k=15)  # Select 15 best features
X_reduced = selector.fit_transform(X, data["Class"])

print(X_reduced.shape)  # Check the reduced dimensionality

# Explore the data
print(data.info())  # General information about the DataFrame
print(data.describe())  # Summary statistics for numerical features
print(data.head())  # View the first few rows

# Check for missing values
print(data.isnull().sum())

# Visualize distributions and relationships
data.hist(figsize=(15, 10))  # Histograms for each feature
plt.show()

plt.scatter(data["V1"], data["Amount"])  # Example scatter plot
plt.show()

plt.matshow(data.corr())  # Correlation matrix
plt.show()
