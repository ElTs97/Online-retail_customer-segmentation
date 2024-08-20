# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import sklearn 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

import warnings
warnings.filterwarnings('ignore')

retail = pd.read_excel('Online Retail.xlsx')
retail.head()

retail.shape
retail.info()
retail.describe()

# Data Cleaning

# calculate % of missing values

retail_missing = round(100*(retail.isnull().sum()/len(retail)),1)
retail_missing

# drop the rows with missing values
retail = retail.dropna()
retail.shape

'''
135280 rows dropped 
'''

# add "value" variable
retail['Value'] = retail['Quantity'] * retail['UnitPrice']

# group by customer ID to find the total of value per customer

#first convert CustomerID to a string
retail['CustomerID'] = retail['CustomerID'].astype(str)
retail['CustomerID'] = retail['CustomerID'].str.replace(r'\.0$', '', regex=True)
retail.info()

grouped = retail.groupby('CustomerID')['Value'].sum().reset_index()
grouped.head()

# count the number of unique invoices per CustomerID
inv_freq = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
inv_freq.head()

# rename columns for better understanding
inv_freq.rename(columns = {'InvoiceNo': 'Freq'}, inplace = True)
inv_freq.head()

# merge datasets
merged_df = pd.merge(grouped, inv_freq, on = 'CustomerID', how = 'inner')
merged_df.head()

# convert InvoiceDate to datetime 
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format = '%d-%m-%Y %H:%M')
retail.head()

# find the min and max date to know the earliest and latest transaction
min_date = min(retail['InvoiceDate'])
max_date = max(retail['InvoiceDate'])
print('Earliest transaction is on: ', min_date, ' and latest transaction is on: ', max_date)

# calculate the date difference between the latest date and each transaction date
retail['date_diff'] = max_date - retail['InvoiceDate']
retail.head()

# calculate the last transaction to find the most recent per user
last_transaction = retail.groupby('CustomerID')['date_diff'].min().reset_index()
last_transaction['date_diff'] = last_transaction['date_diff'].dt.days
last_transaction.head()

# merge dataframes
merged_df = merged_df.merge(last_transaction, on='CustomerID')
merged_df.rename(columns={'date_diff': 'Last_Transaction'}, inplace=True)
merged_df.head()

# OUTLIERS

# create dataframe only with the selected columns
cols = ['Value', 'Freq', 'Last_Transaction']
df = merged_df[cols]

# Melt the dataframe to long format
df_melted = df.melt(var_name='Attributes', value_name='Range')

# Create a box plot with Plotly
fig = px.box(df_melted, x='Attributes', y='Range', 
             labels={'Attributes': 'Attributes', 'Range': 'Range'},
             title="Outliers Variable Distribution")
fig.update_layout(
    xaxis=dict(title="Attributes"),
    yaxis=dict(title="Range"),
    showlegend=False,
    width=800, height=600
)
fig.show()

# remove outliers for Value
Q1 = merged_df.Value.quantile(0.05)
Q3 = merged_df.Value.quantile(0.95)
IQR = Q3 - Q1
merged_df = merged_df[(merged_df.Value >= Q1 - 1.5*IQR) & (merged_df.Value <= Q3 + 1.5*IQR)]

# remove outliers for last transaction
Q1 = merged_df.Last_Transaction.quantile(0.01)
Q3 = merged_df.Last_Transaction.quantile(0.95)
IQR = Q3 - Q1
merged_df = merged_df[(merged_df.Last_Transaction >= Q1 - 1.5*IQR) & (merged_df.Last_Transaction <= Q3 + 1.5*IQR)]

# remove outliers for frequency
Q1 = merged_df.Freq.quantile(0.05)
Q3 = merged_df.Freq.quantile(0.95)
IQR = Q3 - Q1
merged_df = merged_df[(merged_df.Freq >= Q1 - 1.5*IQR) & (merged_df.Freq <= Q3 + 1.5*IQR)]

# create dataframe only with the selected columns
cols = ['Value', 'Freq', 'Last_Transaction']

fig = px.box(merged_df, y=cols, title="Outliers Variable Distribution", 
             labels={'variable': 'Columns', 'value': 'Range'},
             boxmode='group', points='outliers')

fig.update_layout(
    xaxis=dict(title="Columns", title_font=dict(size=14)),
    yaxis=dict(title="Range", title_font=dict(size=14)),
    showlegend=False,
    width=800,
    height=600
)

fig.show()

# rescaling variables

'''
rescaling is important so that they have a comparable scale. Method used: Standarization Scaling
'''

merged_df = merged_df[['Value', 'Freq', 'Last_Transaction']]

scaler = StandardScaler()

scaled_merged_df = scaler.fit_transform(merged_df)
scaled_merged_df.shape

scaled_merged_df = pd.DataFrame(scaled_merged_df)
scaled_merged_df.columns = ['Value', 'Freq', 'Last_Transaction']
scaled_merged_df.head()

'''
Intialiase K-means model. Steps followed:
1. Initialise k points(means) randomly
2. Categorise each item to its closest mean and update the mean's coordinate, which are the averages of the items categorised in that mean so far.
3. Repeat the process for a given number of iterations and create clusters
'''

# k-means with arbitrary k
kmeans = KMeans(n_clusters = 4, max_iter = 50)
kmeans.fit(scaled_merged_df)

kmeans.labels_

# find the optimal number of clusters 

'''
method used: Elbow Curve
'''

ssd = []
range_n_clusters = [2,3,4,5,6,7,8]

for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = num_clusters, max_iter = 50)
    kmeans.fit(scaled_merged_df)
    
    ssd.append(kmeans.inertia_)
    print("For n clusters={0}, the Elbow Method is {1}".format(num_clusters, kmeans.inertia_))
    
fig = px.line(x=range_n_clusters, y = ssd,
              title = 'Elbow Curve Results for K-Means Clustering',
              labels = {'x' : 'Number of Clusters', 'y': 'Sum of Squared Distances (SSD)'})

fig.update_layout(
    xaxis = dict(title_font=dict(size=14)),
    yaxis = dict(title_font=dict(size=14)),
    showlegend = False,
    width = 800,
    height = 600
)

fig.show()

'''
In this case, n=4 or n=5 clusters might be a good choice. n=4 we have a significant reduction from 3 to 4 clusters. n=5 the reduction is still noticable but less dramatic. We want to choose the number of clusters where the rate of decrease starts to diminish significantly, indicating that adding more clusters provides only marginal gains in terms of reduced variance. In this case, n=4 will be used.
'''

# Silhouette Analysis

'''
This method is used to evaluate the quality of clustering by measuring how similar each data point is to its own cluster compared to other clusters, helping to determine the optimal number of clusters and ensuring that the clusters are well-separated and cohesive.
'''

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    # Initialize kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=42)
    kmeans.fit(scaled_merged_df)
    
    cluster_labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_merged_df, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    '''
    The highest silhouette score is achieved with n=3 clusters, indicating that this number provides the best balance between cluster cohesion and separation. 
    The silhouette score tends to decrease with more clusters, which often suggests that additional clusters might be makign the clustering less effective or introdcing more noise. 
    Therefore, 3 clusters will be a strong choice based on the scores, as it appears to offer the best-defined and most distinct groupings for the data.
    '''
    
    '''
    On this analysis both cases of n=3 and n=4 will be tested.
    '''
    
# First Case: n=3 
    
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(scaled_merged_df)
    
kmeans.labels_
    
# assign label
merged_df['Cluster_ID'] = kmeans.labels_
merged_df.head()
    
fig = px.box(merged_df, x ='Cluster_ID', y='Value',
                 title = 'Cluster ID vs Value Box Plot',
                 labels = {'Cluster_ID' : 'Cluster ID'},
                 color='Cluster_ID')
    
fig.update_layout(
        xaxis= dict(title = "Cluster ID", title_font = dict(size=14)),
        yaxis = dict(title = "Value", title_font = dict(size=14)),
        showlegend=False,
        width = 800,
        height = 600
    )
    
fig.show()
    
# cluster ID vs value encoded by frequency
    
fig = px.scatter(merged_df, x='Cluster_ID', y='Value', color = 'Freq',
                     title = "Cluster ID vs Value (encoded by frequency)",
                     labels = {'Cluster_ID' : 'Cluster ID', 'Freq': 'Frequency'})
    
fig.update_layout(
        xaxis = dict(title = "Cluster ID", title_font = dict(size=14)),
        yaxis = dict(title = "Value", title_font = dict(size=14)),
        showlegend = True,
        width = 800,
        height = 600
    )
    
fig.show()
    
# create a custom color palette 

custom_colors = sns.color_palette([
    '#FF6347',
    '#4682B4',
    '#32CD32'
])

# create a plot matrix for each cluster
sns.set(style= "ticks")
sns.pairplot(merged_df, hue = 'Cluster_ID', vars = ['Value', 'Freq'], palette= custom_colors, height = 4, aspect= 1.5)
plt.suptitle('Cluster ID vs Value and Frequency', y = 1.02)
plt.show()

# Cluster ID vs Frequency
fig = px.box(merged_df, x = 'Cluster_ID', y = 'Freq',
             title = "Cluster ID vs Frequency Box Plot",
             labels = {'Cluster_ID': 'Cluster ID', 'Freq': 'Frequency'})

fig.update_layout(
    xaxis = dict(title = 'Cluster ID', title_font = dict(size=14)),
    yaxis = dict(title = 'Frequency', title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

# Cluster ID vs Last Transaction
fig = px.box(merged_df, x = 'Cluster_ID', y = 'Last_Transaction',
             title = 'Cluster ID vs Last Transaction',
             labels = {'Cluster_ID': 'Cluster ID'})

fig.update_layout(
    xaxis = dict(title = "Cluster ID", title_font = dict(size=14)),
    yaxis = dict(title = "Last Transaction", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

# select the two features for clustering
data_for_clustering = merged_df[['Value', 'Freq']]

# specify the number of clusters (n=3)
n_clusters = 3

# apply K-means
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_2D'] = kmeans.fit_predict(data_for_clustering)

# create a scatterplot
fig = px.scatter(merged_df, x = "Value", y="Freq", color = "Cluster_2D",
                 title = 'Clustering by Value and Frequency',
                 labels = {'Freq':'Frequency', 'Cluster_2D' : 'Cluster'})

fig.update_layout(
    xaxis = dict(title = "Value", title_font = dict(size=14)),
    yaxis = dict(title = "Frequency", title_font = dict(size=14)),
    width = 800,
    height = 800
)

fig.show()

# select the 3 features for clustering
data_for_clustering = merged_df[['Value', 'Freq', 'Last_Transaction']]

# apply K-means
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_3D'] = kmeans.fit_predict(data_for_clustering)

# visualise clusters
sns.set(style = "ticks")
sns.pairplot(merged_df, hue = 'Cluster_3D', vars = ['Value', 'Freq', 'Last_Transaction'], palette = 'Set1')
plt.suptitle('Clustering by Value, Frequency and Last Transaction')
plt.show()

# apply k-means clustering 
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_3D'] = kmeans.fit_predict(data_for_clustering)

# create a 3D scatterplot
fig = px.scatter_3d(merged_df, x='Value', y = 'Freq', z = 'Last_Transaction', color = 'Cluster_3D',
                    labels= {'Freq' : 'Frequency', 'Last_Transaction' : 'Last Transaction', 'Cluster_3D': 'Cluster 3D'})

fig.update_layout(
    scene = dict(
        xaxis_title = 'Value',
        yaxis_title = 'Frequency',
        zaxis_title = 'Last Transaction'
        ),
    title = "Clustering by Value, Frequency and Last Transaction",
    width = 800,
    height = 800
)

fig.show()

# Hierarchical clustering 

# Single Linkage 
'''
Single linkage hierarchical clustering is a method where the distance between two clusters is determined by the closest pair of points from each cluster. 
This method can detect elongated or irregularly shaped clusters but can also be sensitive to noise and may produce less compact clusters due to the chaining effect.
'''

mergings = linkage(scaled_merged_df, method = "single", metric = "euclidean")
dendrogram(mergings)
plt.show

# Complete linkage
'''
Complete linkage clustering focuses on the maximum pairwise distance between clusters, which encourages compact and well-separated clusters. 
While it helps avoid the chaining effect seen in single linkage, it can be sensitive to outliers and may form smaller clusters. 
It is especially useful when you want clusters to be tight and compact, and when you are dealing with data that naturally forms spherical clusters.
'''

mergings = linkage(scaled_merged_df, method = "complete", metric = "euclidean")
dendrogram(mergings)
plt.show

# Average linkage
'''
Average linkage clustering is a hierarchical clustering method that calculates the distance between clusters as the average of all pairwise distances between points in the two clusters. 
It offers a balanced approach, avoiding the extremes of very tight clusters (as in complete linkage) or long, chain-like clusters (as in single linkage). 
This method is less sensitive to outliers and works well when you need a compromise between cluster compactness and separation. 
However, it might be computationally intensive for large datasets and may not be the best choice for data with highly irregular cluster shapes.
'''

mergings = linkage(scaled_merged_df, method = "average", metric = "euclidean")
dendrogram(mergings)
plt.show

# cut the dendrogram based on k
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1,)
cluster_labels

# assign cluster labels
merged_df['Cluster_labels'] = cluster_labels
merged_df.head()

fig = px.box(merged_df, x = 'Cluster_labels', y = 'Value',
             title = "Cluster Labels vs Value Box Plot",
             labels = {'Cluster_labels' : 'Cluster labels'})

fig.update_layout(
    xaxis = dict(title = "Cluster Labels", title_font = dict(size=14)),
    yaxis = dict(title = "Value", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

fig = px.box(merged_df, x = 'Cluster_labels', y = 'Freq',
             title = "Cluster Labels vs Frequency Box Plot",
             labels = {'Cluster_labels' : 'Cluster labels'})

fig.update_layout(
    xaxis = dict(title = "Cluster Labels", title_font = dict(size=14)),
    yaxis = dict(title = "Frequency", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

fig = px.box(merged_df, x = 'Cluster_labels', y = 'Last_Transaction',
             title = "Cluster Labels vs Last Transaction Box Plot",
             labels = {'Cluster_labels' : 'Cluster labels'})

fig.update_layout(
    xaxis = dict(title = "Cluster Labels", title_font = dict(size=14)),
    yaxis = dict(title = "Last Transaction", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

# Pairwise scatter plots for clustering (2D)
plt.figure(figsize=(15, 8))

# Amount vs. Frequency
sns.scatterplot(x='Value', y='Freq', hue='Cluster_labels', data=merged_df, palette='Set1')
plt.title('Value vs. Frequency')

# Pairwise scatter plots for clustering (2D)
plt.figure(figsize=(15, 8))  # Adjust the figure size here

# Amount vs. Last Transaction
sns.scatterplot(x='Value', y='Last_Transaction', hue='Cluster_labels', data=merged_df, palette='Set1')
plt.title('Value vs. Last Transaction')

# Pairwise scatter plots for clustering (2D)
plt.figure(figsize=(15, 8))  # Adjust the figure size here

# Frequency vs. Last Transaction
sns.scatterplot(x='Freq', y='Last_Transaction', hue='Cluster_labels', data=merged_df, palette='Set1')
plt.title('Frequency vs. Last Transaction')

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_3D'] = kmeans.fit_predict(data_for_clustering)

# Create a 3D scatter plot matrix with Plotly
fig = px.scatter_3d(merged_df, x='Value', y='Freq', z='Last_Transaction', color='Cluster_3D',
                     labels={'Freq': 'Frequency', 'Last_Transaction': 'Last Transaction', 'Cluster_3D': 'Cluster'})

fig.update_layout(
    scene=dict(
        xaxis_title='Value',
        yaxis_title='Frequency',
        zaxis_title='Last Transaction',
    ),
    title='Clustering by Value, Frequency, and Last Transaction',
    width=800,
    height=800
)

fig.show()

# Second Case: n=4
    
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(scaled_merged_df)
    
kmeans.labels_
    
# assign label
merged_df['Cluster_ID'] = kmeans.labels_
merged_df.head()
    
fig = px.box(merged_df, x ='Cluster_ID', y='Value',
                 title = 'Cluster ID vs Value Box Plot',
                 labels = {'Cluster_ID' : 'Cluster ID'},
                 color='Cluster_ID')
    
fig.update_layout(
        xaxis= dict(title = "Cluster ID", title_font = dict(size=14)),
        yaxis = dict(title = "Value", title_font = dict(size=14)),
        showlegend=False,
        width = 800,
        height = 800
    )
    
fig.show()
    
# cluster ID vs value encoded by frequency
    
fig = px.scatter(merged_df, x='Cluster_ID', y='Value', color = 'Freq',
                     title = "Cluster ID vs Value (encoded by frequency)",
                     labels = {'Cluster_ID' : 'Cluster ID', 'Freq': 'Frequency'})
    
fig.update_layout(
        xaxis = dict(title = "Cluster ID", title_font = dict(size=14)),
        yaxis = dict(title = "Value", title_font = dict(size=14)),
        showlegend = True,
        width = 800,
        height = 800
    )
    
fig.show()
    
# create a custom color palette 

custom_colors = sns.color_palette([
    '#FF6347',
    '#4682B4',
    '#32CD32'
])

# create a plot matrix for each cluster
sns.set(style= "ticks")
sns.pairplot(merged_df, hue = 'Cluster_ID', vars = ['Value', 'Freq'], palette= custom_colors, height = 4, aspect= 1.5)
plt.suptitle('Cluster ID vs Value and Frequency', y = 1.02)
plt.show()

# Cluster ID vs Frequency
fig = px.box(merged_df, x = 'Cluster_ID', y = 'Freq',
             title = "Cluster ID vs Frequency Box Plot",
             labels = {'Cluster_ID': 'Cluster ID', 'Freq': 'Frequency'})

fig.update_layout(
    xaxis = dict(title = 'Cluster ID', title_font = dict(size=14)),
    yaxis = dict(title = 'Frequency', title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

# Cluster ID vs Last Transaction
fig = px.box(merged_df, x = 'Cluster_ID', y = 'Last_Transaction',
             title = 'Cluster ID vs Last Transaction',
             labels = {'Cluster_ID': 'Cluster ID'})

fig.update_layout(
    xaxis = dict(title = "Cluster ID", title_font = dict(size=14)),
    yaxis = dict(title = "Last Transaction", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

# select the two features for clustering
data_for_clustering = merged_df[['Value', 'Freq']]

# specify the number of clusters (n=3)
n_clusters = 4

# apply K-means
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_2D'] = kmeans.fit_predict(data_for_clustering)

# create a scatterplot
fig = px.scatter(merged_df, x = "Value", y="Freq", color = "Cluster_2D",
                 title = 'Clustering by Value and Frequency',
                 labels = {'Freq':'Frequency', 'Cluster_2D' : 'Cluster'})

fig.update_layout(
    xaxis = dict(title = "Value", title_font = dict(size=14)),
    yaxis = dict(title = "Frequency", title_font = dict(size=14)),
    width = 800,
    height = 800
)

fig.show()

# select the 3 features for clustering
data_for_clustering = merged_df[['Value', 'Freq', 'Last_Transaction']]

# apply K-means
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_3D'] = kmeans.fit_predict(data_for_clustering)

# visualise clusters
sns.set(style = "ticks")
sns.pairplot(merged_df, hue = 'Cluster_3D', vars = ['Value', 'Freq', 'Last_Transaction'], palette = 'Set1')
plt.suptitle('Clustering by Value, Frequency and Last Transaction')
plt.show()

# apply k-means clustering 
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_3D'] = kmeans.fit_predict(data_for_clustering)

# create a 3D scatterplot
fig = px.scatter_3d(merged_df, x='Value', y = 'Freq', z = 'Last_Transaction', color = 'Cluster_3D',
                    labels= {'Freq' : 'Frequency', 'Last_Transaction' : 'Last Transaction', 'Cluster_3D': 'Cluster 3D'})

fig.update_layout(
    scene = dict(
        xaxis_title = 'Value',
        yaxis_title = 'Frequency',
        zaxis_title = 'Last Transaction'
        ),
    title = "Clustering by Value, Frequency and Last Transaction",
    width = 800,
    height = 800
)

fig.show()

# Hierarchical clustering 

# Single Linkage 
'''
Single linkage hierarchical clustering is a method where the distance between two clusters is determined by the closest pair of points from each cluster. 
This method can detect elongated or irregularly shaped clusters but can also be sensitive to noise and may produce less compact clusters due to the chaining effect.
'''

mergings = linkage(scaled_merged_df, method = "single", metric = "euclidean")
dendrogram(mergings)
plt.show

# Complete linkage
'''
Complete linkage clustering focuses on the maximum pairwise distance between clusters, which encourages compact and well-separated clusters. 
While it helps avoid the chaining effect seen in single linkage, it can be sensitive to outliers and may form smaller clusters. 
It is especially useful when you want clusters to be tight and compact, and when you are dealing with data that naturally forms spherical clusters.
'''

mergings = linkage(scaled_merged_df, method = "complete", metric = "euclidean")
dendrogram(mergings)
plt.show

# Average linkage
'''
Average linkage clustering is a hierarchical clustering method that calculates the distance between clusters as the average of all pairwise distances between points in the two clusters. 
It offers a balanced approach, avoiding the extremes of very tight clusters (as in complete linkage) or long, chain-like clusters (as in single linkage). 
This method is less sensitive to outliers and works well when you need a compromise between cluster compactness and separation. 
However, it might be computationally intensive for large datasets and may not be the best choice for data with highly irregular cluster shapes.
'''

mergings = linkage(scaled_merged_df, method = "average", metric = "euclidean")
dendrogram(mergings)
plt.show

# cut the dendrogram based on k
cluster_labels = cut_tree(mergings, n_clusters=4).reshape(-1,)
cluster_labels

# assign cluster labels
merged_df['Cluster_labels'] = cluster_labels
merged_df.head()

fig = px.box(merged_df, x = 'Cluster_labels', y = 'Value',
             title = "Cluster Labels vs Value Box Plot",
             labels = {'Cluster_labels' : 'Cluster labels'})

fig.update_layout(
    xaxis = dict(title = "Cluster Labels", title_font = dict(size=14)),
    yaxis = dict(title = "Value", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

fig = px.box(merged_df, x = 'Cluster_labels', y = 'Freq',
             title = "Cluster Labels vs Frequency Box Plot",
             labels = {'Cluster_labels' : 'Cluster labels'})

fig.update_layout(
    xaxis = dict(title = "Cluster Labels", title_font = dict(size=14)),
    yaxis = dict(title = "Frequency", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

fig = px.box(merged_df, x = 'Cluster_labels', y = 'Last_Transaction',
             title = "Cluster Labels vs Last Transaction Box Plot",
             labels = {'Cluster_labels' : 'Cluster labels'})

fig.update_layout(
    xaxis = dict(title = "Cluster Labels", title_font = dict(size=14)),
    yaxis = dict(title = "Last Transaction", title_font = dict(size=14)),
    showlegend = False,
    width = 800,
    height = 800
)

fig.show()

# Pairwise scatter plots for clustering (2D)
plt.figure(figsize=(15, 8))

# Amount vs. Frequency
sns.scatterplot(x='Value', y='Freq', hue='Cluster_labels', data=merged_df, palette='Set1')
plt.title('Value vs. Frequency')

# Pairwise scatter plots for clustering (2D)
plt.figure(figsize=(15, 8))  # Adjust the figure size here

# Amount vs. Last Transaction
sns.scatterplot(x='Value', y='Last_Transaction', hue='Cluster_labels', data=merged_df, palette='Set1')
plt.title('Value vs. Last Transaction')

# Pairwise scatter plots for clustering (2D)
plt.figure(figsize=(15, 8))  # Adjust the figure size here

# Frequency vs. Last Transaction
sns.scatterplot(x='Freq', y='Last_Transaction', hue='Cluster_labels', data=merged_df, palette='Set1')
plt.title('Frequency vs. Last Transaction')

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)
merged_df['Cluster_3D'] = kmeans.fit_predict(data_for_clustering)

# Create a 3D scatter plot matrix with Plotly
fig = px.scatter_3d(merged_df, x='Value', y='Freq', z='Last_Transaction', color='Cluster_3D',
                     labels={'Freq': 'Frequency', 'Last_Transaction': 'Last Transaction', 'Cluster_3D': 'Cluster'})

fig.update_layout(
    scene=dict(
        xaxis_title='Value',
        yaxis_title='Frequency',
        zaxis_title='Last Transaction',
    ),
    title='Clustering by Value, Frequency, and Last Transaction',
    width=800,
    height=800
)

fig.show()