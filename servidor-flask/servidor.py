from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy.ndimage import rotate, center_of_mass, shift, binary_dilation

# Iniciamos el servidor
app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado
rnd_clf_model = joblib.load('random_forest_model.pkl')

def centrar_y_optimizar_imagen(img_2d):
    """
    Toma la imagen dibujada por el usuario, calcula dónde está el centro de masa
    del dibujo y lo mueve exactamente al centro de la cuadrícula (14, 14)
    para que el Random Forest no falle.
    """
    # 1. Crear una versión binaria pura (0 o 255)
    img_binaria = (img_2d > 0).astype(np.uint8) * 255
    
    # 2. Engrosar un poco el trazo usando dilatación (corrige el error de la línea delgada del mouse)
    estructura = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)
    img_dilatada = binary_dilation(img_binaria, structure=estructura).astype(np.uint8) * 255

    # 3. Calcular el centro del dibujo
    cy, cx = center_of_mass(img_dilatada)
    
    # Si el usuario no dibujó nada, regresar matriz vacía
    if np.isnan(cx) or np.isnan(cy):
        return img_dilatada.flatten()
        
    # Calcular cuánto debemos mover el dibujo para que quede en el centro (13.5, 13.5)
    desplazamiento_x = 13.5 - cx
    desplazamiento_y = 13.5 - cy
    
    # Mover físicamente el dibujo al centro
    img_centrada = shift(img_dilatada, shift=[desplazamiento_y, desplazamiento_x], cval=0)
    
    return img_centrada.flatten()

@app.route('/', methods=['POST'])
def predict_react_number():
    if request.method == 'POST':
        react_number = request.get_json() 
        react_number_np = np.array(react_number) 
        
        # Procesamiento geométrico base para corregir la rotación de React
        react_number_reshaped = react_number_np.reshape(28, 28) 
        react_number_flip = np.flip(react_number_reshaped, 0) 
        react_number_rotated = rotate(react_number_flip, angle=-90, reshape=False)
        
        # --- APLICAMOS EL TRUCO DE RECENTRADO Y GROSOR ---
        react_number_clean = centrar_y_optimizar_imagen(react_number_rotated)
        
        # Preparar para el modelo (1 fila, 784 columnas)
        react_number_final = react_number_clean.reshape(1, 784)
        
        # Realizar la predicción
        react_number_prediction = rnd_clf_model.predict(react_number_final)
        
        return f"{react_number_prediction[0]}"
    
    return 'python-server-v2'

if __name__ == '__main__':
    app.run(debug=True)
