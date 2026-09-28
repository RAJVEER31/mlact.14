import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

dt = pd.read_csv('USA_Housing.csv')
dt.head()
dt.info()
dt.describe()
sns.distplot(dt['Price'], bins=30)
x = dt[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = dt['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
cdf

predictions = lm.predict(x_test)

plt.figure(figsize=(10,6))
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')
plt.show()

plt.figure(figsize=(10,6))
sns.distplot((y_test - predictions), bins=50)
plt.title('Distribution of Residuals')
plt.show()

from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
plt.title('Generated Blobs for Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
predicted_clusters = kmeans.fit_predict(data[0])
output = pd.DataFrame(data[0], columns=['Feature 1', 'Feature 2'])
output['Cluster'] = predicted_clusters
plt.scatter(output['Feature 1'], output['Feature 2'], c=output['Cluster'], cmap='rainbow')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
