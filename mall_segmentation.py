#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#load dataset
df = pd.read_csv("Mall_Customers.csv")
print("First 5 rows:\n", df.head())
print("\nInfo:\n", df.info())
#data cleaning
print("\nMissing Values:\n", df.isnull().sum())
df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Spending"}, inplace=True)
#EDA
sns.countplot(data=df, x='Gender')
plt.title("Gender Distribution")
plt.show()
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()
sns.scatterplot(data=df, x="Income", y="Spending", hue="Gender")
plt.title("Income vs Spending")
plt.show()
#feature selection and scaling
X = df[["Income", "Spending"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#elbow method
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("No. of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
#apply KMeans k = 5 based on elbow curve
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
#visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Income', y='Spending', hue='Cluster', palette='Set1', s=100)
plt.title("Customer Segments")
plt.show()
#cluster analysis
print("\nCluster Averages:\n", df.groupby('Cluster')[['Age', 'Income', 'Spending']].mean())
#export results
df.to_csv("segmented_customers.csv", index=False)
print("\nSegmented customer data saved to 'segmented_customers.csv'.")