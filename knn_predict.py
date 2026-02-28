import math

def normalize_dataset(training_data):
    feature_columns = list(zip(*[row[:-1] for row in training_data]))
    min_values = [min(col) for col in feature_columns]
    max_values = [max(col) for col in feature_columns]

    normalized_dataset = []
    for row in training_data:
        normalized_row = [(row[i] - min_values[i]) / (max_values[i] - min_values[i]) for i in range(len(row)-1)]
        normalized_row.append(row[-1])
        normalized_dataset.append(normalized_row)

    return normalized_dataset, min_values, max_values

def euclidean_distance(point_a, point_b):
    return math.sqrt(sum((point_a[i] - point_b[i]) ** 2 for i in range(len(point_a))))

def predict_knn(query_point, training_data, num_neighbors):
    normalized_training_data, min_values, max_values = normalize_dataset(training_data)

    normalized_query_point = [(query_point[i] - min_values[i]) / (max_values[i] - min_values[i]) for i in range(len(query_point))]

    distances = []
    for row in normalized_training_data:
        euclidean_dist = euclidean_distance(normalized_query_point, row[:-1])
        distances.append((euclidean_dist, row[-1]))

    distances.sort(key=lambda distance_label_pair: distance_label_pair[0])
    neighbors = distances[:num_neighbors]
    neighbor_classes = [neighbor[1] for neighbor in neighbors]

    return max(set(neighbor_classes), key=neighbor_classes.count)

dataset = [
    [1.2, 3.1, "A"],
    [2.0, 2.5, "A"],
    [3.5, 0.5, "B"],
    [2.1, 1.0, "B"]
]

print(predict_knn([2.5, 1.5], dataset, 3))
