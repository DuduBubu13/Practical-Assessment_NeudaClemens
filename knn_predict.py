import math

def normalize_dataset(data):
    cols = list(zip(*[d[:-1] for d in data]))
    mins = [min(c) for c in cols]
    maxs = [max(c) for c in cols]

    normalized = []
    for row in data:
        norm = [(row[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(row)-1)]
        norm.append(row[-1])
        normalized.append(norm)

    return normalized, mins, maxs

def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def predict_knn(new_point, data, k):
    normalized_data, mins, maxs = normalize_dataset(data)

    new_norm = [(new_point[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(new_point))]

    distances = []
    for row in normalized_data:
        dist = distance(new_norm, row[:-1])
        distances.append((dist, row[-1]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    classes = [n[1] for n in neighbors]

    return max(set(classes), key=classes.count)

dataset = [
    [1.2, 3.1, "A"],
    [2.0, 2.5, "A"],
    [3.5, 0.5, "B"],
    [2.1, 1.0, "B"]
]

print(predict_knn([2.5, 1.5], dataset, 3))
