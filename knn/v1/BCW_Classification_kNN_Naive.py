from collections import Counter
import math, csv


def knn(data, query, k):
    neighbor_distances_and_indices = []

    for index, entry in enumerate(data):
        distance = euclidean_distance(entry[:-1], query)
        neighbor_distances_and_indices.append((distance, index))

    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    k_nearest_set = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    return k_nearest_distances_and_indices, mode(k_nearest_set)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0

    for index in range(len(point1)):
        sum_squared_distance += math.pow(point1[index] - point2[index], 2)

    return math.sqrt(sum_squared_distance)


def getAccuracy(actualTestSet, predictedSet):
    correctPrediction = 0

    for each in range(len(actualTestSet)):
        if actualTestSet[each][-1] == predictedSet[each][-1]:
            correctPrediction += 1

    return (correctPrediction / float(len(actualTestSet))) * 100.0


def main():
    raw_bcw_trained_data, bcw_trained_data, raw_bcw_test_data, bcw_test_data_result = (
        [],
        [],
        [],
        [],
    )

    with open("bcw-trained_data.csv", "r") as trained_data:
        # Read the trained data into memory
        for each in trained_data.readlines():
            trained_data_sample = each.strip().split(",")
            raw_bcw_trained_data.append(trained_data_sample)

    # Clean/convert the data
    for row in raw_bcw_trained_data:
        data_row = list(map(int, row[1:]))
        bcw_trained_data.append(data_row)

    with open("bcw-data_to_test.csv", "r") as test_data:
        # Read the test data into memory
        for each in test_data.readlines():
            test_data_sample = each.strip().split(",")
            raw_bcw_test_data.append(test_data_sample)

    k_calc = int(round(math.sqrt(len(raw_bcw_test_data))))

    if (k_calc % 2) == 0:
        k_calc -= 1

    # Classify each test sample
    for row in raw_bcw_test_data:
        result_row = row[:-1]
        data_row = list(map(int, row[1:-1]))
        # use the kNN algorithm to classify the data sample
        clf_k_nearest_neighbors, clf_prediction = knn(
            bcw_trained_data,
            data_row,
            k=k_calc,
        )
        result_row.append(str(clf_prediction))
        bcw_test_data_result.append(result_row)

    accuracy = getAccuracy(bcw_test_data_result, raw_bcw_test_data)
    print("Accuracy: " + str("{:.3f}".format(accuracy)) + "%")

    # Export the test results
    with open("bcw_test_data_result.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(bcw_test_data_result)


if __name__ == "__main__":
    main()
