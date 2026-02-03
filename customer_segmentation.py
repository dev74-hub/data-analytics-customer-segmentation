import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_theme(style="darkgrid", palette="mako")
plt.rcParams["figure.figsize"] = (10, 5)

df = pd.read_csv("ifood_df.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

num_cols = df.select_dtypes(include="number").columns


for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

corr = df[num_cols].corr()

sns.heatmap(
    corr,
    cmap="mako",
    annot=False,
    linewidths=0.5
)
plt.title("Correlation Heatmap (iFood Dataset)")
plt.show()

sns.scatterplot(
    x="Income",
    y="MntWines",
    data=df
)
plt.title("Income vs Wine Spending")
plt.show()

sns.scatterplot(
    x="Income",
    y="MntMeatProducts",
    data=df
)
plt.title("Income vs Meat Spending")
plt.show()

sns.scatterplot(
    x="Recency",
    y="NumWebPurchases",
    data=df
)
plt.title("Recency vs Web Purchases")
plt.show()

sns.jointplot(
    x="Income",
    y="MntWines",
    data=df,
    kind="kde",
    fill=True
)
plt.show()

sns.jointplot(
    x="Income",
    y="NumStorePurchases",
    data=df,
    kind="hex"
)
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])

inertia = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sns.scatterplot(
    x="Income",
    y="MntWines",
    hue="Cluster",
    data=df,
    palette="tab10"
)
plt.title("Customer Segmentation: Income vs Wine Spending")
plt.show()

sns.scatterplot(
    x="Recency",
    y="NumWebPurchases",
    hue="Cluster",
    data=df,
    palette="tab10"
)
plt.title("Clusters: Recency vs Web Purchases")
plt.show()


print("\nCluster-wise Mean Values:")
print(df.groupby("Cluster")[num_cols].mean())