import numpy as np
import pandas as pd
from datetime import timedelta as td
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as px

data = pd.read_csv('usercode/customer_segmentation.csv')
print("\n Sample of Dataset")
data.head()

print("Explore Dataset")
data.describe()
data.info()

print(f"Number of purchases: {data['InvoiceNo'].nunique()}")
print(f"Number of customers: {data.CustomerID.nunique()}")

data = data.drop(['index', 'StockCode', 'Description', 'Country'], axis=1)
print(f"\nAfter dropping unncessary columns: {data.head()}")

def PrintMissingCounts(data):
    for col in data.columns.to_list():
        print(f"Missing in column {col}: {data[col].isnull().sum()}")

print("\nProcessing missing column")
PrintMissingCounts(data)
data.dropna(inplace=True)
PrintMissingCounts(data)

data['Price'] = data.UnitPrice * data.Quantity
print("\n After processing price column")
data.head()

# Creating recency DataFrame
purchase_recency = data.groupby("CustomerID", group_keys=True)['InvoiceDate'].agg('max').reset_index(name='Recency').apply(lambda x: x)
print(f"\n Newly created recency dataframe with customer most recent purchase date:\n{purchase_recency.head()}")

purchase_recency.Recency = pd.to_datetime(purchase_recency.Recency)
mostRecentDate = purchase_recency.Recency.max() + td(days=1)

purchase_recency.Recency = purchase_recency.Recency.apply(lambda x: (mostRecentDate - x).days)
print(f"\nProcessed recency dataframe: \n{purchase_recency.head()}")

# Creating frequency DataFrame
purchase_frequency = data.groupby('CustomerID', group_keys=True)['CustomerID'].count().reset_index(name='Frequency').apply(lambda x: x)
print(f"\nNewly created and processed frequency DateFrame:\n {purchase_frequency.head()}")

# Creating monetary value per customer
purchase_amounts = data.groupby('CustomerID', group_keys=True)['Price'].sum().reset_index(name='Monetary').apply(lambda x: x)
print(f"\nNewly created and processed monetary DateFrame:\n {purchase_amounts.head()}")

# Preparing data for RFM analysis
rfm_data = purchase_recency.merge(purchase_frequency, on='CustomerID').merge(purchase_amounts, on='CustomerID')
print(f"\nNewly created DateFrame for RFM anaylsis:\n {rfm_data.head()}")

#rfm_data_copy = rfm_data.copy()

# Preprocess RFM data for ML
num_cols = ['Recency', 'Frequency', 'Monetary']
preprocessor = StandardScaler()
rfm_data[num_cols] = preprocessor.fit_transform(rfm_data[num_cols])
print(f"\nPreprocessed RFM data for ML:\n {rfm_data.head()}")

x = pd.DataFrame(rfm_data[num_cols])
x.head()

# Finding optimal number of cluser for KMeans analysis
wcsse = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters, max_iter=100, random_state=42)
    kmeans.fit(x)
    wcsse.append(kmeans.inertia_)

print(wcsse)

plot = px.Figure(data=[px.Scatter(x=list(range(1, 11)), y=wcsse, mode='lines+markers')])
plot.update_layout(
    title="Finding K with the Elbow Technique",
    xaxis_title="Value of K",
    yaxis_title="Within Cluster Sum of Squared Distances",
)
plot.show()

# Cluster the data
k = 3
finalKMeans = KMeans(n_clusters=k, max_iter = 100, n_init = 10, random_state = 0)
y_clusters = finalKMeans.fit_predict(x)

print(f"\n Raw cluster data: \n{np.unique(y_clusters, return_counts=True)}")

# Explore the clusters
rfm_data['Clusters'] = y_clusters

dfs = []
for i in range(k+1):
    dfs.append(rfm_data[rfm_data.Clusters == i])
    print(f"Customer dataframe for cluster {i} \n {dfs[i].head()} ")

# Visualize the clusters
graphes = []
for i_cluster in range(k+1):
    d=dfs[i_cluster]
    graph = px.Scatter3d(
        x = d.Recency,
        y = d.Frequency,
        z = d.Monetary,
        text=d.CustomerID,
        name = "Cluster " + str(i_cluster))
    graphes.append(graph)

layout = px.Layout (
        margin=dict(l=0, r=0, b=30, t=30), title='Customer Segmentation',
        scene = dict(
            xaxis = dict (title = "Recency"),
            yaxis = dict (title = "Frequency"),
            zaxis = dict (title = "Amount Spent")
        )
    )
fig = px.Figure(data=graphes, layout=layout)

# Update the data point labels
fig.update_traces( hovertemplate=" Customer ID:%{text} <br> Recency:%{x} <br> Frequency:%{y} <br> Amount Spent:%{z} ")
fig.show()