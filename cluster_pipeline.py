import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Loading the dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully\n")

print(df.head())


# Checking missing values
print("\nMissing Values:\n")
print(df.isnull().sum())


# Removing missing values if any
df = df.dropna()


# Selecting important columns
features = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]

X = df[features]

print("\nSelected Features:\n")
print(X.head())


# Scaling the data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)


# Finding best K value using Elbow Method
wcss = []

K = range(2, 11)

for k in K:

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(scaled_data)

    wcss.append(model.inertia_)


# Plotting Elbow Graph
plt.figure(figsize=(8,5))

plt.plot(K, wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.savefig("elbow_graph.png")

plt.show()


# Training K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(scaled_data)


# Adding cluster column
df['Cluster'] = clusters


# Evaluating the model
score = silhouette_score(
    scaled_data,
    clusters
)

print("\nModel Evaluation")
print("----------------------")

print("Silhouette Score :", round(score, 2))


# Simple interpretation
if score > 0.5:
    print("Clusters are properly separated")

elif score > 0.3:
    print("Clusters are reasonably separated")

else:
    print("Clusters are not clearly separated")


# Visualizing customer clusters
plt.figure(figsize=(10,6))

sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2',
    s=80
)

plt.title("Customer Segmentation using K-Means")

plt.savefig("customer_clusters.png")

plt.show()


# Cluster Summary
cluster_summary = df.groupby('Cluster')[features].mean()

print("\nCluster Summary:\n")

print(cluster_summary)


# Saving final output
df.to_csv(
    "customer_segmented_output.csv",
    index=False
)

print("\nProject Completed Successfully!")
