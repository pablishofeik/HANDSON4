import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            predicted_label = np.argmax(np.bincount(nearest_labels))
            predictions.append(predicted_label)
        return predictions

# Conjunto de datos 
dataset = [
    [158, 58, 'M'],
    [158, 63, 'M'],
    [160, 59, 'M'],
    [160, 60, 'M'],
    [163, 60, 'M'],
    [163, 61, 'M'],
    [160, 64, 'L'],
    [163, 64, 'L'],
    [165, 61, 'L'],
    [165, 62, 'L'],
    [165, 65, 'L'],
    [168, 62, 'L'],
    [168, 63, 'L'],
    [168, 66, 'L'],
]

# Convertir dataset a formato numpy
dataset = np.array(dataset)

# Separar características y etiquetas
X = dataset[:, :-1].astype(float)
y = dataset[:, -1]

# Convertir etiquetas a valores numéricos
class_mapping = {'M': 0, 'L': 1}
y_numeric = np.array([class_mapping[label] for label in y])

# Crear instancia de KNN
knn = KNN(k=3)

# Entrenar el modelo
knn.fit(X, y_numeric)

X_test = np.array([[163, 61], [168, 66]])

predictions = knn.predict(X_test)

predictions_labels = np.array(['M' if pred == 0 else 'L' for pred in predictions])

# Imprimir predicciones
print("Predicciones:", predictions_labels)