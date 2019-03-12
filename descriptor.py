import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import cv2
from sklearn.cluster import DBSCAN


def hist_com(comparison_histogram, labels):
    array_length = range(len(comparison_histogram))
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0
    for i in array_length:
        for j in array_length:
            hist_1 = np.float32(comparison_histogram[i, :])
            hist_2 = np.float32(comparison_histogram[j, :])
            match_dist = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
            if match_dist < 0.3:
                evaluator = True
            else:
                evaluator = False
            if (labels[i] == labels[j]) and (evaluator == True):
                true_pos = true_pos + 1
            elif (labels[i] != labels[j]) and (evaluator == True):
                false_pos = false_pos + 1
            elif (labels[i] == labels[j]) and (evaluator == False):
                false_neg = false_neg + 1
            elif (labels[i] != labels[j]) and (evaluator == False):
                true_neg = true_neg + 1

    pre = true_pos / (true_pos + false_pos)
    rec = true_pos / (true_pos + false_neg)
    f1_score = 2 * ((pre * rec) / (pre + rec))
    return pre, rec, f1_score


def score_cal(descriptor, cluster_label):
    sample_size = range(len(descriptor))
    cluster_no = np.nanmax(descriptor)  # Number of clusters
    true_positive = 0
    total_pos = 0
    no_arr = []
    model_arr = np.bincount(descriptor)
    for i in cluster_no:
        no_arr.clear()
        for j in sample_size:
            if cluster_label[i] == cluster_label[j]:
                no_arr.append(descriptor[j])  # Real model number of each element in the cluster
        freq_arr = np.bincount(no_arr)  # Frequency of each model number in the cluster
        max_val = freq_arr.argmax()  # Most frequent(dominant) model number in the cluster
        true_positive = true_positive + freq_arr[max_val + 1]  # Number of the most frequent element in the cluster
        false_negative = false_negative + (model_arr[max_val + 1] - true_positive)
        total_pos = total_pos + true_positive + np.sum(np.delete(freq_arr, [max_val + 1]))  # Total positives

    sensitivity = true_positive / total_pos
    specificity = true_positive / (true_positive + false_negative)
    score_f = (2 * (sensitivity * specificity)) / (sensitivity + specificity)
    return sensitivity, specificity, score_f


if __name__ == '__main__':

    label_cluster = []
    score_cluster = []
    dist_arr = []

    descriptor_values = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/esf/DescriptorValuesESF.csv')
    descriptor_labels = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/esf/ClusterResult2.csv')
    descriptor_values = descriptor_values.drop(descriptor_values.columns[640], axis=1)
    descriptor_labels = descriptor_labels.iloc[:, 3]

    # mds = manifold.MDS(n_components=23, dissimilarity="euclidean")
    # mds_descriptor = mds.fit(descriptor_values).embedding_
    # mds_descriptor = np.abs(mds_descriptor)

    """pca = PCA(n_components=39)
    ratio = pca.fit(descriptor_values.T).explained_variance_ratio_
    cum_ratio = sum(ratio)
    components = pca.fit(descriptor_values.T).components_
    components = np.abs(components.T)
    print('explained variance ratio: ' + repr(ratio))
    print('\ncumulative explained variance ratio: ' + repr(cum_ratio))"""

    # k_means = KMeans(n_clusters=39, init='random')
    # label_cluster = k_means.fit(descriptor_values).labels_

    # clustering = AgglomerativeClustering(n_clusters=39)
    # label_cluster = clustering.fit(mds_descriptor).labels_

    # db_scan = DBSCAN(eps=3, min_samples=2).fit(components)
    # label_cluster = db_scan.labels_
    # n_clusters_ = len(set(label_cluster))
    # print('no. of clusters: ' + repr(n_clusters_))

    descriptor_values = descriptor_values.to_numpy()
    descriptor_labels = descriptor_labels.to_numpy()
    descriptor_labels = descriptor_labels.astype(np.int64)

    precision, recall, f_score = hist_com(descriptor_values, descriptor_labels)
    # precision, recall, f_score = score_cal(descriptor_labels, label_cluster)

    print('Precision: ' + repr(precision) + '\nRecall: ' + repr(recall) + '\nF-1 Score: ' + repr(f_score))
    print('\nEND')



