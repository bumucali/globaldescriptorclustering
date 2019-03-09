import pandas
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

if __name__ == '__main__':

    label_cluster = []
    score_cluster = []
    true_pos = 0
    total_pos = 0
    true_neg = 0
    false_neg = 0

    descriptor_values = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/mppf/DescriptorValuesMPPF.csv')
    descriptor_labels = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/mppf/ClusterResult2.csv')
    descriptor_values = descriptor_values.drop(descriptor_values.columns[384], axis=1)
    descriptor_labels = descriptor_labels.iloc[:, 3]

    # mds = manifold.MDS(n_components=39, dissimilarity="euclidean")
    # mds_descriptor = mds.fit(descriptor_values).embedding_

    """pca = PCA(n_components=39)
    ratio = pca.fit(descriptor_values.T).explained_variance_ratio_
    cum_ratio = sum(ratio)
    components = pca.fit(descriptor_values.T).components_
    components = components.T
    print('explained variance ratio: ' + repr(ratio))
    print('\ncumulative explained variance ratio: ' + repr(cum_ratio))"""

    k_means = KMeans(n_clusters=23, init='random')
    label_cluster = k_means.fit(descriptor_values).labels_
    # score_cluster.append(k_means.fit(descriptor_values).score(descriptor_values))

    # clustering = AgglomerativeClustering(n_clusters=23)
    # label_cluster = clustering.fit(components).labels_
    sample_size = range(len(label_cluster))

    """descriptor_values = descriptor_values.values

    for i in range(len(descriptor_values)):
        for j in range(len(descriptor_values)):
            hist_1 = descriptor_values[i, :]
            hist_2 = descriptor_values[j, :]
            hist_size = 640
            mean_1 = np.mean(hist_1)
            mean_2 = np.mean(hist_2)
            first_term = 1 / (math.sqrt(mean_1 * mean_2 * hist_size * hist_size))
            second_term = np.sum(np.sqrt(np.multiply(hist_1, hist_2)))
            match_dist = math.sqrt(abs(1 - (first_term * second_term)))
            # print('dist: ' + repr(match_dist))
            if match_dist < 0.3:
                total_pos = total_pos + 1
                if descriptor_labels[i] == descriptor_labels[j]:
                    true_pos = true_pos + 1
            elif (match_dist >= 0.3) and (descriptor_labels[i] == descriptor_labels[j]):
                false_neg = false_neg + 1

    precision = true_pos / total_pos
    recall = true_pos / (true_pos + false_neg)
    f_score = (2 * (precision * recall)) / (precision + recall)

    print('total positive: ' + repr(total_pos) + '\nfalse negative: ' + repr(false_neg) + '\ntrue positive: ' +
          repr(true_pos))
    print('Precision: ' + repr(precision) + '\nRecall: ' + repr(recall) + '\nF-1 Score: ' + repr(f_score))
    print('\nEND')"""

    for i in sample_size:
        for j in sample_size:
            if (label_cluster[i] == label_cluster[j]) and (j > i):
                total_pos = total_pos + 1
                if descriptor_labels[i] == descriptor_labels[j]:
                    true_pos = true_pos + 1
            elif (label_cluster[i] != label_cluster[j]) and (descriptor_labels[i] != descriptor_labels[j]) and (j > i):
                true_neg = true_neg + 1

    total_pair = (len(label_cluster) * (len(label_cluster) - 1)) / 2
    false_neg = (total_pair - total_pos) - true_neg
    precision = true_pos / total_pos
    recall = true_pos / (true_pos + false_neg)
    f_score = (2 * (precision * recall)) / (precision + recall)

    print('total positive: ' + repr(total_pos) + '\nfalse negative: ' + repr(false_neg) + '\ntrue positive: ' +
          repr(true_pos))
    print('Precision: ' + repr(precision) + '\nRecall: ' + repr(recall) + '\nF-1 Score: ' + repr(f_score))
    print('\nEND')
