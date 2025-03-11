import tensorflow as tf
from tensorflow import keras

# Charger MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation et reshaping
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)  # ✅ Correct format
x_test = x_test.reshape(-1, 28, 28, 1)

# Construire le modèle CNN
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),  # 🔹 Utiliser Input() au lieu de `input_shape=`
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiler et entraîner
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Sauvegarde propre du modèle
model.save("mnist.h5")
print("✅ Modèle réentraîné et sauvegardé correctement sous mnist.h5")
