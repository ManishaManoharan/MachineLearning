import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import pickle

print("Downloading dataset...")

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X = X / 255.0

# Use only small portion to train faster
X_train = X[:10000]
y_train = y[:10000]

X_test = X[10000:12000]
y_test = y[10000:12000]

print("Training MLP model...")

mlp = MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation='relu',
    solver='adam',
    max_iter=20
)

mlp.fit(X_train, y_train)

print("Training Accuracy:", mlp.score(X_train, y_train))
print("Test Accuracy:", mlp.score(X_test, y_test))

pickle.dump(mlp, open("mlp_model.pkl","wb"))

print("Model saved!")