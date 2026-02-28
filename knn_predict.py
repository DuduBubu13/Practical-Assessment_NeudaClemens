import math

def normalize_dataset(data):
    cols = list(zip(*[row[:-1] for row in data]))
    mins = [min(column) for column in cols]
    maxs = [max(column) for column in cols]

    normalized = []
    for row in data:
        normalized_row = [(row[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(row)-1)]
        normalized_row.append(row[-1])
        normalized.append(normalized_row)

    return normalized, mins, maxs

def euclidean_distance(point_a, point_b):
    return math.sqrt(sum((point_a[i] - point_b[i]) ** 2 for i in range(len(point_a))))

def predict_knn(new_point, data, k):
    normalized_data, mins, maxs = normalize_dataset(data)

    normalized_new_point = [(new_point[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(new_point))]

    distances = []
    for row in normalized_data:
        dist = euclidean_distance(normalized_new_point, row[:-1])
        distances.append((dist, row[-1]))

    distances.sort(key=lambda distance_label_pair: distance_label_pair[0])
    neighbors = distances[:k]
    classes = [neighbor[1] for neighbor in neighbors]

    return max(set(classes), key=classes.count)

dataset = [
    [1.2, 3.1, "A"],
    [2.0, 2.5, "A"],
    [3.5, 0.5, "B"],
    [2.1, 1.0, "B"]
]

print(predict_knn([2.5, 1.5], dataset, 3))
