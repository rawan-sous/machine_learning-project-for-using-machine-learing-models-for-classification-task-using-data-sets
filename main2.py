import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.sparse import issparse

# Load the CSV file
file_path = 'creditcard.csv' 
df = pd.read_csv(file_path)

# Check for missing data
missing_data = df.isnull().sum()
# Display the count of missing values for each column
print("Missing Data:")
print(missing_data) 

# Select only the columns V1 to V28 for correlation analysis
selected_columns = df.loc[:, 'V1':'V28']

# Calculate correlation matrix
correlation_matrix = selected_columns.corr()

# Create a heatmap to visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap (V1 to V28)')
plt.show()

# Identify highly correlated features
threshold = 0.8  # Set your correlation threshold
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

# Exclude 'id' and 'Class' from dropping
highly_correlated_features -= {'id', 'Class', 'Amount'}

# Display highly correlated features
print("Highly Correlated Features:", list(highly_correlated_features))

# Use SelectKBest to choose the best features based on ANOVA F-value
k_best = 20  # Set the number of best features to select
feature_selector = SelectKBest(f_classif, k=k_best)

# Fit the feature selector to the data
X = selected_columns
y = df['Class']  # Assuming 'Class' is the target variable
X_best = feature_selector.fit_transform(X, y)

# Get the selected features
selected_features = X.columns[feature_selector.get_support()]

# Display the selected features
print("Selected Features:", list(selected_features))

# Drop highly correlated and unselected features from the DataFrame
features_to_drop = highly_correlated_features.union(set(X.columns) - set(selected_features))
df_filtered = df.drop(columns=features_to_drop)

# Calculate correlation matrix for the filtered DataFrame
filtered_correlation_matrix = df_filtered.corr()

# Create a heatmap to visualize correlations of the filtered DataFrame
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Filtered Correlation Heatmap')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = 'creditcard.csv' 
df = pd.read_csv(file_path)

# Count the number of records in the original dataset
original_records_count = len(df)
print("Original Records Count:", original_records_count)

# Assuming 'Class' is the target variable
strata_cols = ['Class']
strata_values = df[strata_cols]

# Define the subset size
subset_size = 5000

# Stratified sampling to maintain the distribution of the target variable
df_subset, _ = train_test_split(df, test_size=(original_records_count - subset_size) / original_records_count, stratify=strata_values, random_state=42)
print("Subset Records Count:", len(df_subset))

# Continue with your analysis or modeling using the df_subset DataFrame
# ...

# Optionally, you can save the subset to a new CSV file
subset_file_path = 'creditcard_subset.csv'
df_subset.to_csv(subset_file_path, index=False)
print(f"Subset saved to: {subset_file_path}")

import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming 'Class' is the target variable
X = df_subset.drop('Class', axis=1)
y = df_subset['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model with k=1
knn_model_1 = KNeighborsClassifier(n_neighbors=1)
knn_model_1.fit(X_train, y_train)
y_pred_1 = knn_model_1.predict(X_test)

# Evaluate the model with k=1
accuracy_1 = accuracy_score(y_test, y_pred_1)
print("Baseline Model (k=1) - Accuracy:", accuracy_1)

# Report additional metrics for k=1
if len(np.unique(y_test)) > 1:
    print("Classification Report (k=1):")
    print(classification_report(y_test, y_pred_1))
    print("Confusion Matrix (k=1):")
    print(confusion_matrix(y_test, y_pred_1))

# Baseline model with k=3
knn_model_3 = KNeighborsClassifier(n_neighbors=3)
knn_model_3.fit(X_train, y_train)
y_pred_3 = knn_model_3.predict(X_test)

# Evaluate the model with k=3
accuracy_3 = accuracy_score(y_test, y_pred_3)
print("\nBaseline Model (k=3) - Accuracy:", accuracy_3)

# Report additional metrics for k=3
if len(np.unique(y_test)) > 1:
    print("\nClassification Report (k=3):")
    print(classification_report(y_test, y_pred_3))
    print("Confusion Matrix (k=3):")
    print(confusion_matrix(y_test, y_pred_3))


