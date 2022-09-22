## chapter 2-1

Hierarchical clustering of the grain data

```python
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

```
![image](https://user-images.githubusercontent.com/109948144/191679128-73a10614-eef2-49b8-a908-b8d6315a4966.png)
<br><br/>

## chapter 2-2

Hierarchies of stocks

```python
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()

```
![image](https://user-images.githubusercontent.com/109948144/191679190-fd03553c-e756-4989-a5ba-5ac98b16825f.png)
<br><br/>

## chapter 2-3

Different linkage, different hierarchical clustering!

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()

```
![image](https://user-images.githubusercontent.com/109948144/191679234-d84d2e84-3ca8-4634-a17d-05a9d3327f28.png)
<br><br/>

## chapter 2-4

Extracting the cluster labels

```python
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

```

## chapter 2-5

t-SNE visualization of grain dataset

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()

```
![image](https://user-images.githubusercontent.com/109948144/191679381-4968d798-e12f-4f33-814b-333f1f62588a.png)
<br><br/>

## chapter 2-6

A t-SNE map of the stock market

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

```
![image](https://user-images.githubusercontent.com/109948144/191679500-b80ef698-6b0a-4348-beba-7129f35926cc.png)
