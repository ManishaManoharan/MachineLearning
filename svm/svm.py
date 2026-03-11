import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = SVC(kernel='rbf')
model.fit(X_train_scaled, y_train)
X_plot = iris.data[:, :2]
y_plot = iris.target

X_plot_train, _, y_plot_train, _ = train_test_split(
    X_plot, y_plot, test_size=0.3, random_state=42
)

scaler_plot = StandardScaler()
X_plot_train_scaled = scaler_plot.fit_transform(X_plot_train)

model_plot = SVC(kernel='rbf')
model_plot.fit(X_plot_train_scaled, y_plot_train)

def predict_species():
    try:
        f1 = float(entry1.get())
        f2 = float(entry2.get())
        f3 = float(entry3.get())
        f4 = float(entry4.get())

        user_input = np.array([[f1, f2, f3, f4]])
        user_input = scaler.transform(user_input)

        prediction = model.predict(user_input)
        species = iris.target_names[prediction[0]]

        result_label.config(text="Predicted Species: " + species)

    except:
        messagebox.showerror("Error", "Please enter valid numeric values")

def show_plot():
    plt.figure()
    
    x_min, x_max = X_plot_train_scaled[:, 0].min() - 1, X_plot_train_scaled[:, 0].max() + 1
    y_min, y_max = X_plot_train_scaled[:, 1].min() - 1, X_plot_train_scaled[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    Z = model_plot.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_plot_train_scaled[:, 0], X_plot_train_scaled[:, 1], c=y_plot_train)

    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.title("SVM Decision Boundary (First 2 Features)")

    plt.show()

root = tk.Tk()
root.title("SVM Iris Classifier")
root.geometry("400x400")

title = tk.Label(root, text="Iris Flower Prediction (SVM)", font=("Arial", 14))
title.pack(pady=10)

tk.Label(root, text="Sepal Length (cm)").pack()
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root, text="Sepal Width (cm)").pack()
entry2 = tk.Entry(root)
entry2.pack()

tk.Label(root, text="Petal Length (cm)").pack()
entry3 = tk.Entry(root)
entry3.pack()

tk.Label(root, text="Petal Width (cm)").pack()
entry4 = tk.Entry(root)
entry4.pack()

predict_button = tk.Button(root, text="Predict", command=predict_species)
predict_button.pack(pady=10)

plot_button = tk.Button(root, text="Show Decision Boundary", command=show_plot)
plot_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack()

root.mainloop()