from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from time import time

# Takes a pandas dataframe containing the cluster assignment and ground truth for each data point
# and returns the purity of the cluster results
def clustering_purity(cluster_results, index_to_name):
    clusters = cluster_results['cluster'].unique()
    m = cluster_results.shape[0]

    # Purity for each cluster
    cluster_purities = []
    cluster_sizes = []
    most_common_classes = []
    for j in clusters:
        cluster_j = cluster_results[cluster_results['cluster'] == j]
        m_j = cluster_j.shape[0]
        cluster_sizes.append(m_j)
        classes = cluster_j['class'].unique()

        # Class probability distribution for this cluster
        class_probabilities = []
        for i in classes:
            cluster_j_class_i = cluster_j[cluster_j['class'] == i]
            m_ij = cluster_j_class_i.shape[0]
            class_probabilities.append(m_ij / m_j)

        # Calculate cluster purity
        cluster_purity = np.max(np.array(class_probabilities))
        cluster_purities.append(cluster_purity)

        # Save most common class per cluster
        most_common_classes.append(index_to_name[class_probabilities.index(cluster_purity)])

    total_purity = 0
    for i, size in enumerate(cluster_sizes):
        total_purity += (size / m) * cluster_purities[i]

    # Pandas dataframe containing per cluster results
    results_table = pd.DataFrame({'cluster': clusters,
                                  'cluster_size': cluster_sizes,
                                  'most_common_class': most_common_classes,
                                  'purity': cluster_purities,
                                  'total_purity': total_purity})

    return total_purity, results_table

# Takes a pandas dataframe containing the cluster assignment and ground truth for each data point
# and returns the entropy of the cluster results
def clustering_entropy(cluster_results, index_to_name):
    clusters = cluster_results['cluster'].unique()
    m = cluster_results.shape[0]

    # Entropy for each cluster
    cluster_entropies = []
    cluster_sizes = []
    most_common_classes = []
    for j in clusters:
        cluster_j = cluster_results[cluster_results['cluster'] == j]
        m_j = cluster_j.shape[0]
        cluster_sizes.append(m_j)
        classes = cluster_j['class'].unique()

        # Class probability distribution for this cluster
        class_probabilities = []
        for i in classes:
            cluster_j_class_i = cluster_j[cluster_j['class'] == i]
            m_ij = cluster_j_class_i.shape[0]
            class_probabilities.append(m_ij/m_j)

        # Calculate cluster entropy
        cluster_entropy = 0
        for p in class_probabilities:
            cluster_entropy -= p * np.log2(p)
        cluster_entropies.append(cluster_entropy)

        # Save most common class per cluster
        most_common_classes.append(index_to_name[class_probabilities.index(np.max(np.array(class_probabilities)))])

    total_entropy = 0
    for i, size in enumerate(cluster_sizes):
        total_entropy += (size / m) * cluster_entropies[i]

    # Pandas dataframe containing per cluster results
    results_table = pd.DataFrame({'cluster': clusters,
                                  'cluster_size': cluster_sizes,
                                  'most_common_class': most_common_classes,
                                  'entropy': cluster_entropies,
                                  'total_entropy': total_entropy})

    return total_entropy, results_table



def main():
    model_name = 'encoder_caltech256.h5'

    encoder = load_model(model_name)

    encode_datagen = ImageDataGenerator(rescale=1. / 255)
    predict_generator = encode_datagen.flow_from_directory(
        'data/256_ObjectCategories',
        target_size=(128, 128),
        batch_size=1,
        class_mode='input', shuffle=False)

    n_images = 29780

    # Encode all images
    encoded_imgs = encoder.predict_generator(predict_generator, n_images, verbose=1)

    # Flatten encoded images to create feature vector for clustering
    encoded_imgs_feature_vecs = encoded_imgs.reshape(n_images, 8 * 8 * 600)

    # Perform K-means clustering on flattened feature vector
    print('Starting K-means..')
    t0 = time()
    kmeans = KMeans(n_clusters=256, n_init=2, n_jobs=-1)
    clusters = kmeans.fit_predict(encoded_imgs_feature_vecs)
    duration = time() - t0
    print("done in %fs" % (duration))
    print()

    # Prepare data for evaluation functions
    cluster_results = pd.DataFrame({'cluster': clusters, 'class': predict_generator.classes})

    # Save cluster results
    cluster_results.to_csv(model_name[:-3] + 'cluster_results.csv', index=False)

    class_index_to_name = {v: k for k, v in predict_generator.class_indices.items()}

    print('Evaluating entropy..')
    t0 = time()
    total_entropy, entropy_per_cluster = clustering_entropy(cluster_results, index_to_name=class_index_to_name)
    duration = time() - t0
    print("done in %fs" % (duration))
    print()

    print('Evaluating purity..')
    total_purity, purity_per_cluster = clustering_purity(cluster_results, index_to_name=class_index_to_name)
    duration = time() - t0
    print("done in %fs" % (duration))
    print()

    print('Entropy:')
    print(str(total_entropy))
    print(entropy_per_cluster.to_string())

    print('\n\n\nPurity: ')
    print(str(total_purity))
    print(purity_per_cluster.to_string())

    entropy_per_cluster.to_csv(model_name[:-3] + 'entropy_details.csv', index=False)
    purity_per_cluster.to_csv(model_name[:-3] + 'purity_details.csv', index=False)



if __name__ == '__main__':
    main()