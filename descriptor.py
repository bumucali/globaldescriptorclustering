from typing import Any, Union

import pandas
import pylab as pl

from sklearn.cluster import KMeans

if __name__ == '__main__':

    label_cluster = []
    score_cluster = []
    true_pos = 0
    total_pos = 0
    false_neg = 0

    descriptor_values = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/esf/DescriptorValuesESF.csv')
    descriptor_labels = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/esf/ClusterResult2.csv')
    descriptor_values = descriptor_values.drop(descriptor_values.columns[640], axis=1)
    descriptor_labels = descriptor_labels.iloc[:, 3]

    k_means = KMeans(n_clusters=39, init='random')
    label_cluster = k_means.fit(descriptor_values).labels_
    score_cluster.append(k_means.fit(descriptor_values).score(descriptor_values))

    for i in range(len(label_cluster)):
        for j in range(len(descriptor_labels)):
            if (label_cluster[i] == label_cluster[j]) and (j > i):
                total_pos = total_pos + 1
                if descriptor_labels[i] == descriptor_labels[j]:
                    true_pos = true_pos + 1
            elif (label_cluster[i] != label_cluster[j]) and (j > i) and (descriptor_labels[i] == descriptor_labels[j]):
                false_neg = false_neg + 1

    precision = true_pos / total_pos
    recall = true_pos / (true_pos + false_neg)
    f_score = (2 * (precision * recall)) / (precision + recall)

    print('Precision: ' + repr(precision) + '\nRecall: ' + repr(recall) + '\nF-1 Score: ' + repr(f_score))
    print('\nEND')