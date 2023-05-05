# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Read the file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url)

data.columns = ["age", "workclass", "education", "fnlwgt", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "captial-loss", "hours-per-week", "native-country", "SalaryCat"]

# Remove un wanted data
data = data.dropna()
data = data[~data.isin(['?']).any(axis=1)]
data = data.drop_duplicates()

# Convert Categorical data to Numerical data
categorical_columns = ['workclass', 'education', 'fnlwgt', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'SalaryCat']

for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])

data = data.apply(pd.to_numeric, errors='coerce')

data = data.dropna()


selected_columns = ['education-num', 'age']
data_selected = data[selected_columns]

scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_selected)

k_values = range(2,15)
sse_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_standardized)
    sse_values.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_values, sse_values, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method for Optimal K')
plt.show()

silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_standardized)
    label = kmeans.labels_
    score = silhouette_score(data_standardized, label)
    silhouette_scores.append(score)

plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for different values of k')
plt.show()

# Find the optimal k value
optimal_k = k_values[np.argmax(silhouette_scores)]
print("Optiaml K value:", optimal_k)

# Perform k-means clustering using the calculated optimal k value
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_standardized)
label = kmeans.labels_

# Plot cluster
plt.scatter(data_selected['education-num'],data_selected['age'], c=label, marker='o', edgecolors='none')
plt.xlabel('Education Level')
plt.ylabel('age')
plt.title('K-Means Clustering with Optimal K')
plt.legend()
plt.show()

# Add cluster labels to original DataFrame
data_selected['cluster'] = label

# Describe each cluster
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    cluster_data = data_selected[data_selected['cluster'] == i]
    print("Number of Obervation:", len(cluster_data))
    print("Mean age:", np.mean(cluster_data['age']))
    print("Mean education level:", np.mean(cluster_data['education-num']))


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_standardized)
label = kmeans.labels_

# Plot cluster
plt.scatter(data_selected['education-num'],data_selected['age'], c=label, marker='o', edgecolors='none')
plt.xlabel('Education Level')
plt.ylabel('age')
plt.title('K-Means Clustering with K = 2')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_standardized)
label = kmeans.labels_

# Plot cluster
plt.scatter(data_selected['education-num'],data_selected['age'], c=label, marker='o', edgecolors='none')
plt.xlabel('Education Level')
plt.ylabel('age')
plt.title('K-Means Clustering with K = 3')
plt.legend()
plt.show()
