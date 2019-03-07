import pandas
import pylab as pl

from sklearn.cluster import KMeans

if __name__ == '__main__':

    label_cluster = []
    score_cluster = []
    descriptor_values = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/esf/DescriptorValuesESF.csv')
    descriptor_values = descriptor_values.drop(descriptor_values.columns[640], axis=1)
    cluster_range = range(50, 200)

    for cluster_size in cluster_range:
        k_means = KMeans(n_clusters=cluster_size, init='random')
        print(k_means.fit(descriptor_values).labels_)
        score_cluster.append(k_means.fit(descriptor_values).score(descriptor_values))

    pl.plot(cluster_range, score_cluster)

    pl.xlabel('Number of Clusters')

    pl.ylabel('Score')

    pl.title('Elbow Curve')

    pl.show()

    """
    Nc = range(35, 50)

    k_means = [KMeans(n_clusters=i, init='random') for i in Nc]

    score = [k_means[i].fit(descriptor_values).score(descriptor_values) for i in range(len(k_means))]

    pl.plot(Nc, score)

    pl.xlabel('Number of Clusters')

    pl.ylabel('Score')

    pl.title('Elbow Curve')

    pl.show()

    """

print('END')
