import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import binary_dilation, rotate # <-- Herramientas de mejora visual

# 1. Carga del dataset MNIST usando Keras
print("Cargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocesamiento base (Mantenemos 0 y 255 para React)
x_train_binary = (x_train > 0).astype(np.uint8) * 255
x_test_binary = (x_test > 0).astype(np.uint8) * 255

# --- ESTRATEGIA DE POTENCIACIÓN REAL ---
print("Engrosando y aumentando el set de datos para adaptarlo al dibujo de la Web...")
x_train_augmented = []
y_train_augmented = []

# Creamos una máscara en cruz para engrosar el trazo de 1px a un formato natural
estructura_grosor = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)

for img, label in zip(x_train_binary, y_train):
    # a) Guardamos la imagen con el grosor corregido
    img_dilated = binary_dilation(img, structure=estructura_grosor).astype(np.uint8) * 255
    x_train_augmented.append(img_dilated.flatten())
    y_train_augmented.append(label)
    
    # b) Multiplicamos los datos: Añadimos una versión rotada 10 grados a la izquierda
    img_rot_izq = rotate(img_dilated, angle=10, reshape=False)
    x_train_augmented.append(img_rot_izq.flatten())
    y_train_augmented.append(label)

    # c) Añadimos una versión rotada 10 grados a la derecha
    img_rot_der = rotate(img_dilated, angle=-10, reshape=False)
    x_train_augmented.append(img_rot_der.flatten())
    y_train_augmented.append(label)

# Convertimos a arrays de numpy (Pasamos de 60,000 muestras a 180,000 muestras robustas)
x_train_final = np.array(x_train_augmented)
y_train_final = np.array(y_train_augmented)

# Procesamos el set de prueba igual (aplanado y engrosado)
x_test_final = np.array([binary_dilation(img, structure=estructura_grosor).flatten() for img in x_test_binary])

# 3. Visualización para que veas cómo la IA verá el número ahora
plt.imshow(x_train_final[0].reshape(28, 28), cmap='binary')
plt.title(f"Así procesará el modelo optimizado (Label: {y_train_final[0]})")
plt.axis('off')
plt.show()

# 4. Configuración Profesional del RandomForest
print(f"\nIniciando entrenamiento masivo con {x_train_final.shape[0]} muestras...")
rnd_clf = RandomForestClassifier(
    n_estimators=300,       # 300 árboles para máxima estabilidad decisional
    max_depth=None,         
    min_samples_split=4,    # Ajustado sutilmente para el nuevo volumen de datos
    min_samples_leaf=2,     
    n_jobs=-1,              # Explota al máximo todos los hilos de tu CPU
    random_state=42,        
    verbose=1               
)

# 5. Entrenamiento
rnd_clf.fit(x_train_final, y_train_final)

# 6. Evaluación
print("\n--- EVALUACIÓN EN CONSOLA ---")
y_pred = rnd_clf.predict(x_test_final)
accuracy = accuracy_score(y_test, y_pred)
print(f" NUEVA PRECISIÓN REAL: {accuracy * 100:.2f}%")

# 7. Guardado (Reemplaza el archivo anterior automáticamente)
model_filename = 'random_forest_model.pkl'
joblib.dump(rnd_clf, model_filename)
print(f"\n ¡Modelo de alta potencia guardado como '{model_filename}'!")
