if __name__ == '__main__':

    label_cluster = []
    true_pos = 0
    total_pos = 0
    true_neg = 0
    descriptor_labels = [2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 3, 2, 3, 3, 3, 2]
    label_cluster = [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]
    sample_size = range(len(label_cluster))

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