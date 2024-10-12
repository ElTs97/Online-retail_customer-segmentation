# Online-retail_customer-segmentation


This project analyzes an online retail dataset to understand customer purchasing behavior and segment customers based on their purchasing patterns. The analysis includes data cleaning, feature engineering, exploratory data analysis (EDA), K-Means clustering, and hierarchical clustering.

## Technologies Used

- **Python**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **Plotly**: For interactive visualizations.
- **Scikit-learn**: For machine learning and data preprocessing.
- **SciPy**: For scientific and technical computing.
- **Jupyter Notebook**: For interactive computing.

## Dataset

The dataset used in this analysis is the "Online Retail" dataset, which contains transactions occurring between 2010 and 2011 for a UK-based online retailer. Key features include:

- **InvoiceNo**: Unique identifier for each invoice
- **StockCode**: Unique identifier for each product
- **Description**: Description of the product
- **Quantity**: Quantity of each product purchased
- **InvoiceDate**: Date of the invoice
- **UnitPrice**: Price per unit
- **CustomerID**: Unique identifier for each customer

## Analysis Steps

### Data Cleaning

The analysis begins with cleaning the dataset, calculating the percentage of missing values, and dropping any rows with missing data. A new variable, **Value**, was created by multiplying **Quantity** by **UnitPrice** to represent the total transaction value for each purchase.

### Feature Engineering

The dataset is processed to group customer transactions by `CustomerID`, calculating total purchase value, unique invoice counts, and the date of the last transaction for each customer.

### Outlier Detection

Key metrics—**Value**, **Frequency**, and **Last Transaction**—are explored to identify and visualize outliers. This step ensures that the subsequent analysis and modeling are not adversely affected by extreme values.

### Clustering Analysis

#### K-Means Clustering

1. **Rescaling Variables**: Standardizing the key metrics to ensure they are on a comparable scale for effective clustering.

2. **Applying K-Means**: The K-Means algorithm is applied to segment customers into clusters based on their purchasing behavior. Different numbers of clusters were evaluated:
   - **Case 1**: n=3
   - **Case 2**: n=4

3. **Visualizing Clusters**: Various visualizations are created to illustrate the relationship between clusters and key metrics:
   - Box plots comparing **Value** and **Frequency** against **Cluster IDs**.
   - Pair plots to visualize distributions and relationships among **Value**, **Frequency**, and **Last Transaction**.
   - 2D scatter plots to visualize clusters based on two dimensions: **Value** and **Frequency**.
   - 3D scatter plots to visualize clusters based on three dimensions: **Value**, **Frequency**, and **Last Transaction**.

#### Hierarchical Clustering

1. **Single, Complete, and Average Linkage**: Hierarchical clustering is performed using different linkage methods to assess cluster quality and structure.

2. **Dendrogram Visualization**: Dendrograms are plotted to visualize the hierarchical relationships between clusters, aiding in understanding cluster compactness and separation.

3. **Cutting the Dendrogram**: The dendrogram is cut to assign cluster labels, which are then used for further analysis.

4. **Box Plots for Cluster Labels**: Box plots are created to compare clusters against **Value**, **Frequency**, and **Last Transaction**, similar to the K-Means approach.

### Results and Insights

The K-Means clustering identified distinct customer segments for both cases (n=3 and n=4), while hierarchical clustering confirmed similar patterns with a different approach. Key insights include:
- Customers in different clusters exhibit varying spending behaviors, purchase frequencies, and recency of last transactions.
- Visualizations provide a clear distinction between customer segments, allowing for targeted marketing strategies.

## Future Work

Future analysis could involve:
- Further exploration of identified customer segments for personalized marketing strategies.
- Additional predictive modeling to forecast customer behavior and enhance customer retention efforts.
