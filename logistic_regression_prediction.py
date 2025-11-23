import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

z = 1.72
prob = sigmoid(z)

print("Probability:", prob)

if prob >= 0.5:
    print("Class: 1")
else:
    print("Class: 0")
