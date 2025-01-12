#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data: Customer purchase history
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AnnualIncome': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Select features for clustering
X = df[['AnnualIncome', 'SpendingScore']]

# Create a KMeans instance with k clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Predict the clusters for each customer
df['Cluster'] = kmeans.predict(X)

# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()

# Print the resulting DataFrame with cluster assignments
print(df)

