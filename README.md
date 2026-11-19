# Reconocimiento de Números y Prendas

Proyecto de visión por computadora con dos módulos: un reconocedor de números dibujados a mano (MNIST, puerto 5000) y un reconocedor de prendas que clasifica imágenes reales en 10 categorías mediante transfer learning con MobileNetV2 (puerto 5001). Los modelos se sirven con Flask y se consumen desde una interfaz en React con pestañas.

## Tecnologías
TensorFlow · Flask · React · MobileNetV2 · Python

## Dataset
Las imágenes de entrenamiento no se incluyen en el repositorio por su tamaño. Se usa el dataset de Kaggle "Fashion Product Images (Small)" y se colocan en `servidor-flask/images_ia/`.

Desarrollado en SENATI, Ingeniería de Software con IA.

## Nota sobre el modelo Random Forest
El archivo `random_forest_model.pkl` (~199 MB) supera el límite de GitHub y no se incluye. Se genera ejecutando `modelo.py` en `servidor-flask/`.
