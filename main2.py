import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

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
