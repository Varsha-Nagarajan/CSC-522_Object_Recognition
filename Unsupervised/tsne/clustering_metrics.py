import numpy as np
import pandas as pd

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
