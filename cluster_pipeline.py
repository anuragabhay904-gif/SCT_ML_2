# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())


#  Check Missing Values
print(df.isnull().sum())
# Remove missing values
df = df.dropna()


# Select Important Features
# Using multiple factors for better clustering

features = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]

X = df[features]

print("\nSelected Features:")
print(X.head())

# -----------------------------------
# Step 4: Scale Features
# -----------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(X)

# -----------------------------------
# Step 5: Find Best K Value
# -----------------------------------

wcss = []

K = range(2, 11)

for k in K:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(scaled_data)

    wcss.append(kmeans.inertia_)

# -----------------------------------
# Step 6: Save Elbow Graph
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(K, wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.savefig("elbow_graph.png")

plt.show()

# -----------------------------------
# Step 7: Train K-Means Model
# -----------------------------------

# Beginner-friendly choice

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(scaled_data)

# Add cluster column
df['Cluster'] = clusters

# -----------------------------------
# Step 8: Evaluate Model
# -----------------------------------

score = silhouette_score(
    scaled_data,
    clusters
)

print("\nModel Evaluation")
print("----------------------")

print("Silhouette Score:", round(score, 2))

# Simple interpretation

if score > 0.5:
    print("Clusters are well separated.")
elif score > 0.3:
    print("Clusters are reasonably separated.")
else:
    print("Clusters have weak separation.")

# -----------------------------------
# Step 9: Save Cluster Visualization
# -----------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2',
    s=80
)

plt.title("Customer Segmentation")

plt.savefig("customer_clusters.png")

plt.show()

# -----------------------------------
# Step 10: Cluster Summary
# -----------------------------------

cluster_summary = df.groupby('Cluster')[features].mean()

print("\nCluster Summary:")
print(cluster_summary)

# -----------------------------------
# Step 11: Save Final Output
# -----------------------------------

df.to_csv(
    "customer_segmented_output.csv",
    index=False
)

print("\nProject Completed Successfully!")