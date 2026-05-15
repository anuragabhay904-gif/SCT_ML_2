## Customer Segmentation Using K-Means Clustering
This repository contains a machine learning pipeline that segments retail store customers based on their purchase history. By identifying distinct customer groups, businesses can optimize targeted marketing, improve inventory management, and enhance customer retention strategies.
## 🚀 Features

* Data Preprocessing: Handles missing values, removes outliers using the Interquartile Range (IQR) method, and scales features via StandardScaler.
* RFM Feature Engineering: Extracts Recency, Frequency, and Monetary value from raw transaction logs.
* Optimal Cluster Selection: Utilizes the Elbow Method and Silhouette Coefficient to determine the ideal number of customer segments ($K$).
* K-Means Clustering: Groups customers based on behavioral similarities.
* Data Visualization: Includes interactive 2D/3D scatter plots, radar charts for cluster profiling, and distribution plots using Seaborn and Plotly.

## 📦 Tech Stack

* Language: Python 3.10+
* Data Libraries: NumPy, Pandas
* Machine Learning: Scikit-Learn
* Visualization: Matplotlib, Seaborn, Plotly

## 📊 Customer Segments Identified

* VIP Customers: High frequency, high spending, recent purchases.
* Churn Risks: High historical spending, but no recent purchases.
* Frugal Loyalists: Frequent shoppers with low average order value.
* New Shippers: Recent buyers with low frequency and low spending.

## 🛠️ Getting Started## Prerequisites

pip install numpy pandas scikit-learn matplotlib seaborn plotly

## Usage

   1. Place your transaction data in the data/ directory as transactions.csv.
   2. Run the main segmentation pipeline:

python src/cluster_pipeline.py

