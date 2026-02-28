import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

linear_combination = 1.72
predicted_probability = sigmoid(linear_combination)

print("Probability:", predicted_probability)

if predicted_probability >= 0.5:
    print("Class: 1")
else:
    print("Class: 0")
