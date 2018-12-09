from __future__ import print_function
## importing the required packages
import pandas as pd
from datetime import datetime as dt
import numpy as np
from Unsupervised.tsne import clustering_metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import (manifold, decomposition)
from sklearn.cluster import KMeans

np.random.seed(7)
from tensorflow import set_random_seed

set_random_seed(7)
from keras.preprocessing.image import ImageDataGenerator

# print(device_lib.list_local_devices())

print("Import done at ", str(dt.now()))

target = list()
images = list()
# batch_size = 10000
batch_size = 29780
img_x, img_y = 128, 128


def generate_samples(generator):
    for Sample_X, Sample_Y in generator:
        data = Sample_X
        return data


train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    directory="data/256_ObjectCategories",
    target_size=(img_x, img_y),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical', seed=42)

print("Generator initialized at ", str(dt.now()))

X = generate_samples(train_generator)
idx = np.random.choice(29780, 10000, replace=False)
X = X[idx]

print("Samples generated at ", str(dt.now()))

n_samples, n_features, *rest = X.shape
n_neighbors = 30


def plot_embedding(tsne_output, color):
    plt.scatter(tsne_output[:, 0], tsne_output[:, 1], c=color)
    plt.show()


def plot_embedding_3d(tsne_output, color, axx):
    axx.scatter(tsne_output[:, 0], tsne_output[:, 1], tsne_output[:, 2], c=color)
    pyplot.show()


print("Flattening started at ", str(dt.now()))
X_flat = X.reshape(10000, 128 * 128 * 3)
print("Flattening ended at ", str(dt.now()))

# Computing PCA
print("PCA started at ", str(dt.now()))
X_pca = decomposition.TruncatedSVD(n_components=75).fit_transform(X_flat)
print("PCA ended at ", str(dt.now()))

# Computing t-SNE
print("TSNE started at ", str(dt.now()))
tsne = manifold.TSNE(n_components=2, verbose=2, init='pca', random_state=0)
print("TSNE ended at ", str(dt.now()))
print("TSNE fit started at ", str(dt.now()))
X_tsne = tsne.fit_transform(X_pca)
print("TSNE fit ended at ", str(dt.now()))
# print("Plot started at ", str(dt.now()))
# plot_embedding(X_tsne, train_generator.classes[:batch_size])
# print("Plot ended at ", str(dt.now()))
print("Output started at ", str(dt.now()))
np.savetxt('output/small.txt', X_tsne)
print("Output ended at ", str(dt.now()))

print("KMeans started at ", str(dt.now()))
kmeans = KMeans(n_clusters=256, n_init=10, n_jobs=-1)
# other = kmeans.fit_predict(X_tsne)
y_kmeans = kmeans.fit(X_tsne).predict(X_tsne)
centers = kmeans.cluster_centers_
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=train_generator.classes[idx])
plot_embedding(centers, 'black')
np.savetxt('output/clustering.txt', y_kmeans)
print("KMeans ended at ", str(dt.now()))

print("Entropy evaluation started at ", str(dt.now()))
cluster_results = pd.DataFrame({'cluster': y_kmeans, 'class': train_generator.classes[idx]})
class_index_to_name = {v: k for k, v in train_generator.class_indices.items()}
total_entropy, entropy_per_cluster = clustering_metrics.clustering_entropy(cluster_results, index_to_name=class_index_to_name)
print("Entropy evaluation ended at ", str(dt.now()))

print("Purity evaluation started at ", str(dt.now()))
total_purity, purity_per_cluster = clustering_metrics.clustering_purity(cluster_results, index_to_name=class_index_to_name)
print("Purity evaluation ended at ", str(dt.now()))

print("Entropy: ")
print(str(total_entropy))
print(str(entropy_per_cluster.to_string()))

print("Purity: ")
print(str(total_purity))
print(str(purity_per_cluster.to_string()))