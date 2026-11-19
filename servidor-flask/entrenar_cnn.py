import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Carga del dataset MNIST usando Keras
print("Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocesamiento Binario para mantener compatibilidad con tu React (0 o 23)
# Una CNN trabaja mejor con valores flotantes, por lo que convertiremos a 0.0 y 1.0.
# ¡No te preocupes! En tu servidor Flask solo dividiremos los datos recibidos entre 23.
x_train_binary = (x_train > 0).astype(np.float32)
x_test_binary = (x_test > 0).astype(np.float32)

# Las CNN necesitan una 4ta dimensión para los canales de color (1 porque es escala de grises)
x_train_cnn = np.expand_dims(x_train_binary, axis=-1)
x_test_cnn = np.expand_dims(x_test_binary, axis=-1)

# 3. Arquitectura de la CNN (Estructura estándar de alto rendimiento)
model = models.Sequential([
    # Primera capa convolucional: Extrae bordes y líneas básicas
    layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same', input_shape=(28, 28, 1)),
    layers.AveragePooling2D((2, 2)),
    
    # Segunda capa convolucional: Combina líneas para detectar curvas y formas
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
    
    
    # Tercera capa convolucional: Detecta estructuras complejas del número
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Aplanamos los mapas de características 2D a un vector 1D
    layers.Flatten(),
    
    # Capa densa (neuronas completamente conectadas) con Dropout para evitar sobreajuste
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Capa de salida: 10 neuronas (una para cada dígito del 0 al 9)
    layers.Dense(10, activation='softmax')
])

# 4. Compilación del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Muestra la estructura de la red en consola

# 3. Entrenamiento de la CNN
print("\nIniciando el entrenamiento de la CNN...")
# 3 épocas son más que suficientes para alcanzar un rendimiento brutal (~99%)
history = model.fit(x_train_cnn, y_train, epochs=3, batch_size=64, validation_data=(x_test_cnn, y_test))

# 6. Evaluación con datos de prueba
test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=2)
print(f"\nPRECISIÓN FINAL DE LA CNN EN PRUEBA: {test_acc * 100:.2f}%")

# 7. Guardar el modelo en formato nativo de Keras (.keras)
model_filename = 'cnn_number_model.keras'
model.save(model_filename)
print(f"Modelo CNN guardado exitosamente como '{model_filename}'")
