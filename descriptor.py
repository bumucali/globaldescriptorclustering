import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
from sklearn.decomposition import PCA
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

TEST_SPLIT = 688
PATH = '/home/berkay/Desktop/EndDescriptors/mppf/39_Classes/'


def hist_com(comparison_histogram, labels):
    array_length = range(len(descriptor_labels))
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0
    for i in array_length:
        for j in array_length:
            hist_1 = np.float32(comparison_histogram[i, :])
            hist_2 = np.float32(comparison_histogram[j, :])
            match_dist = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
            if match_dist < 0.28:
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
    f1_score_1 = 2 * ((pre * rec) / (pre + rec))
    return pre, rec, f1_score_1


def score_cal(descriptor, cluster_label):
    sample_size = range(len(descriptor))
    cluster_no = np.nanmax(descriptor)  # Number of clusters
    true_positive = 0
    total_pos = 0
    false_negative = 0
    print('shape: ' + repr(descriptor.shape))
    no_list = []
    model_arr = np.bincount(descriptor)
    for i in range(cluster_no):
        no_list.clear()
        for j in sample_size:
            if i == cluster_label[j]:
                no_list.append(descriptor[j])  # Real model number of each element in the cluster
        no_arr = np.asarray(no_list)
        freq_arr = np.bincount(no_arr)  # Frequency of each model number in the cluster
        max_val = freq_arr.argmax()  # Most frequent(dominant) model number in the cluster
        true_positive = true_positive + freq_arr[max_val]  # Number of the most frequent element in the cluster
        false_negative = false_negative + (model_arr[max_val] - freq_arr[max_val])
        total_pos = total_pos + len(no_arr)  # Total positives

    sensitivity = true_positive / total_pos
    specificity = true_positive / (true_positive + false_negative)
    score_f = (2 * (sensitivity * specificity)) / (sensitivity + specificity)
    return sensitivity, specificity, score_f


def support_vector(values, labels):
    # data_set = np.concatenate(descriptor_values, descriptor_labels, axis=1)
    # data_set = np.random.shuffle(data_set)
    string_svm = 'confusion_svm.csv'

    x_train = values[:TEST_SPLIT, :]
    y_train = labels[:TEST_SPLIT]
    x_test = values[TEST_SPLIT:, :]
    y_test = labels[TEST_SPLIT:]
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 10]}
    svm_grid = GridSearchCV(SVC(), param_grid, verbose=1, cv=3, iid=True)

    # svm = SVC(C=1.0, kernel='rbf', gamma='auto')
    svm_grid.fit(x_train, y_train)
    y_prediction = svm_grid.predict(x_test)
    print("Best Parameters:\n", svm_grid.best_params_)
    print("Best Estimators:\n", svm_grid.best_estimator_)
    print(accuracy_score(y_test, y_prediction))
    precision_1 = precision_score(y_test, y_prediction, average='micro')
    recall_1 = recall_score(y_test, y_prediction, average='micro')
    f_score_1 = f1_score(y_test, y_prediction, average='micro')
    conf_arr = confusion_matrix(y_test, y_prediction)
    np.savetxt(PATH + string_svm, conf_arr, delimiter=',')

    return precision_1, recall_1, f_score_1


def decision_tree(x_values, y_labels):
    string_tree = 'confusion_tree.csv'

    x_train = x_values[:TEST_SPLIT, :]
    y_train = y_labels[:TEST_SPLIT]
    x_test = x_values[TEST_SPLIT:, :]
    y_test = y_labels[TEST_SPLIT:]
    tree_classifier = DecisionTreeClassifier(criterion='gini')

    tree_classifier.fit(x_train, y_train)
    y_prediction = tree_classifier.predict(x_test)
    print(accuracy_score(y_test, y_prediction))
    precision_tree = precision_score(y_test, y_prediction, average='micro')
    recall_tree = recall_score(y_test, y_prediction, average='micro')
    f_score_tree = f1_score(y_test, y_prediction, average='micro')
    conf_mat = confusion_matrix(y_test, y_prediction)
    np.savetxt(PATH + string_tree, conf_mat, delimiter=',')

    return precision_tree, recall_tree, f_score_tree


def random_forest_calc(values_x, labels_y):
    string_forest = 'confusion_forest.csv'

    x_training = values_x[:TEST_SPLIT, :]
    y_training = labels_y[:TEST_SPLIT]
    x_testing = values_x[TEST_SPLIT:, :]
    y_testing = labels_y[TEST_SPLIT:]
    random_clf = RandomForestClassifier(n_estimators=100)

    random_clf.fit(x_training, y_training)
    y_pred = random_clf.predict(x_testing)
    print(accuracy_score(y_testing, y_pred))
    precision_rf = precision_score(y_testing, y_pred, average='micro')
    recall_rf = recall_score(y_testing, y_pred, average='micro')
    f_score_rf = f1_score(y_testing, y_pred, average='micro')
    confusion_arr = confusion_matrix(y_testing, y_pred)
    np.savetxt(PATH + string_forest, confusion_arr, delimiter=',')

    return precision_rf, recall_rf, f_score_rf


def pca_calculation(desc_value):
    pca = PCA(n_components=39)
    ratio = pca.fit(desc_value.T).explained_variance_ratio_
    cum_ratio = sum(ratio)
    components = pca.fit(desc_value.T).components_
    components = np.abs(components.T)
    print('explained variance ratio: ' + repr(ratio))
    print('\ncumulative explained variance ratio: ' + repr(cum_ratio))

    return components


def mds_calculation(descriptors):
    mds = manifold.MDS(n_components=39, dissimilarity="euclidean")
    mds_descriptor = mds.fit(descriptors).embedding_
    mds_descriptor = np.abs(mds_descriptor)

    return mds_descriptor


def k_clustering(value_to_cluster):
    k_means = KMeans(n_clusters=39)
    label_from_cluster = k_means.fit(value_to_cluster).labels_

    return label_from_cluster


def agg_clustering(cluster_values):
    clustering = AgglomerativeClustering(n_clusters=39)
    label_cluster_agg = clustering.fit(cluster_values).labels_

    return label_cluster_agg


if __name__ == '__main__':

    label_cluster = []
    descriptor_path = 'DescriptorValues2.csv'
    full_path = PATH + descriptor_path

    descriptor_values = pandas.read_csv(full_path)
    descriptor_labels = descriptor_values[['Label']]
    descriptor_values = descriptor_values.drop(descriptor_values.columns[384], axis=1)
    print(descriptor_values.head(2))

    descriptor_values = descriptor_values.to_numpy()
    descriptor_labels = descriptor_labels.to_numpy()
    descriptor_labels = descriptor_labels.astype(np.int64)
    descriptor_labels = descriptor_labels.ravel()

    # mds_values = mds_calculation(descriptor_values)

    # pca_comp = pca_calculation(descriptor_values)

    # label_cluster = k_clustering(pca_comp)

    # label_cluster = agg_clustering(pca_comp)

    # precision, recall, f_score = hist_com(mds_values, descriptor_labels)
    # precision, recall, f_score = score_cal(descriptor_labels, label_cluster)

    # precision, recall, f_score = support_vector(descriptor_values, descriptor_labels)
    # precision, recall, f_score = decision_tree(descriptor_values, descriptor_labels)
    precision, recall, f_score = random_forest_calc(descriptor_values, descriptor_labels)

    print('Precision: ' + repr(precision) + '\nRecall: ' + repr(recall) + '\nF-1 Score: ' + repr(f_score))
    print('\nEND')
