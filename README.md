# MiniProjecClustering
Clustering using K-means Algorithm . Clustering based on the IQ Test of Government Schools Students And Private school Students.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Dataset
data = {
    'School_Type': ['Govt','Govt','Govt','Govt','Govt','Private','Private','Private','Private','Private'],
    'IQ_Score': [82, 90, 95, 88, 92, 105, 110, 115, 108, 112]
}

df = pd.DataFrame(data)

# Step 2: Convert school type to numeric
df['School_Type_Num'] = df['School_Type'].map({'Govt':0, 'Private':1})

# Step 3: Standardize
features = df[['School_Type_Num','IQ_Score']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Step 4: K-Means
kmeans = KMeans(n_clusters=2, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Visualization
plt.scatter(df['IQ_Score'], df['Cluster'])
plt.xlabel("IQ Score")
plt.ylabel("Cluster Group")
plt.title("Clustering of Students Based on IQ Score")
plt.show()

# Step 6: Summary
print(df)
print("\nCluster-wise Average IQ:\n", df.groupby('Cluster')['IQ_Score'].mean())

