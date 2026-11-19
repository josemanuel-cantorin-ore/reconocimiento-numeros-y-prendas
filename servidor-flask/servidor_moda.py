import base64
import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "images_ia" / "myntradataset"
IMAGENES_DIR = BASE_DIR / "images_ia" / "images"
MODELO_PATH = BASE_DIR / "modelo_moda.keras"
CLASES_PATH = BASE_DIR / "clases_moda.json"
TARGET_SIZE = (224, 224)

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}},
)


def normalizar_clase(article_type):
    """Normaliza los tipos reales del dataset a una taxonomía segura y balanceada."""
    valor = str(article_type).strip()
    texto = valor.lower()

    if any(token in texto for token in ["tshirt", "t-shirt", "top", "tops"]):
        return "Tshirts"
    if any(token in texto for token in ["shirt", "shirts"]):
        return "Shirts"
    if any(token in texto for token in ["jean", "jeans", "pant", "pants", "trouser", "trousers", "track pant", "track pants", "legging", "leggings"]):
        return "Pants"
    if any(token in texto for token in ["short", "shorts"]):
        return "Shorts"
    if any(token in texto for token in ["dress", "dresses", "saree", "sarees", "skirt", "skirts"]):
        return "Dresses"
    if any(token in texto for token in ["kurta", "kurtas"]):
        return "Kurtas"
    if any(token in texto for token in ["casual shoes", "formal shoes", "flats", "heels", "sandals", "flip flops"]):
        return "Casual Shoes"
    if any(token in texto for token in ["sports shoes", "sneakers", "running shoes"]):
        return "Sports Shoes"
    if any(token in texto for token in ["handbag", "handbags", "backpack", "backpacks", "wallet", "wallets"]):
        return "Handbags"
    if any(token in texto for token in ["watch", "watches", "sunglasses", "belt", "belts", "perfume", "socks"]):
        return "Accessories"
    return None


def localizar_styles():
    for nombre in ("styles.txt", "styles.csv"):
        ruta = DATASET_DIR / nombre
        if ruta.exists():
            return ruta
    return None


def cargar_clases():
    if CLASES_PATH.exists():
        with CLASES_PATH.open("r", encoding="utf-8") as archivo:
            return json.load(archivo)
    return [
        "Shirts",
        "Tshirts",
        "Pants",
        "Shorts",
        "Dresses",
        "Kurtas",
        "Casual Shoes",
        "Sports Shoes",
        "Handbags",
        "Accessories",
    ]


CLASES = cargar_clases()
CLASES_ES = {
    "Shirts": "Camisa",
    "Tshirts": "Camiseta",
    "Pants": "Pantalón",
    "Shorts": "Shorts",
    "Dresses": "Vestido",
    "Kurtas": "Kurta",
    "Casual Shoes": "Zapatos casuales",
    "Sports Shoes": "Zapatillas deportivas",
    "Handbags": "Bolsos",
    "Accessories": "Accesorios",
}

try:
    MODELO = tf.keras.models.load_model(MODELO_PATH) if MODELO_PATH.exists() else None
except Exception:
    MODELO = None


def etiqueta_clase(nombre_clase):
    """Convierte los nombres internos del modelo a etiquetas legibles para la UI."""
    return CLASES_ES.get(str(nombre_clase), str(nombre_clase))


def cargar_muestras_dataset():
    style_path = localizar_styles()
    if style_path is None:
        return []

    try:
        df = pd.read_csv(style_path, on_bad_lines="skip")
        df.columns = [str(col).strip() for col in df.columns]
        if "articleType" not in df.columns:
            return []

        df = df[["id", "articleType"]].copy()
        df["articleType"] = df["articleType"].map(normalizar_clase)
        df = df[df["articleType"].isin(CLASES)].copy()
        df["ruta_imagen"] = df["id"].astype(str).map(lambda value: IMAGENES_DIR / f"{value}.jpg")
        df = df[df["ruta_imagen"].map(lambda ruta: ruta.exists())].copy()
        if df.empty:
            return []

        muestras = []
        for _, fila in df.head(200).iterrows():
            muestras.append({
                "id": int(fila["id"]),
                "articleType": fila["articleType"],
                "ruta_imagen": str(fila["ruta_imagen"]),
            })
        return muestras
    except Exception:
        return []


def obtener_muestras_activas():
    """Devuelve todas las imágenes válidas del dataset ordenadas de forma estable."""
    archivos = sorted(IMAGENES_DIR.glob("*.jpg")) + sorted(IMAGENES_DIR.glob("*.jpeg")) + sorted(IMAGENES_DIR.glob("*.png"))
    archivos = sorted({ruta.resolve(): ruta for ruta in archivos}.values(), key=lambda ruta: ruta.name)

    if not archivos:
        return []

    mapa_clases = {}
    style_path = localizar_styles()
    if style_path is not None:
        try:
            df = pd.read_csv(style_path, on_bad_lines="skip")
            if "id" in df.columns and "articleType" in df.columns:
                for _, fila in df[["id", "articleType"]].dropna().iterrows():
                    try:
                        clase = normalizar_clase(fila["articleType"])
                        if clase in CLASES:
                            mapa_clases[int(float(fila["id"]))] = clase
                    except Exception:
                        continue
        except Exception:
            mapa_clases = {}

    muestras = []
    for ruta in archivos:
        try:
            muestra_id = int(ruta.stem)
        except ValueError:
            continue

        clase_real = mapa_clases.get(muestra_id)
        if clase_real is None:
            continue

        muestras.append({
            "id": muestra_id,
            "articleType": clase_real,
            "ruta_imagen": str(ruta),
        })

    return sorted(muestras, key=lambda item: item["id"])[:200]


def obtener_muestras_galeria(cantidad=12):
    """Devuelve una selección aleatoria y distinta para la galería sin romper la validación de IDs."""
    muestras = obtener_muestras_activas()
    if len(muestras) <= cantidad:
        return muestras

    seleccion = muestras[:]
    random.shuffle(seleccion)
    return seleccion[:cantidad]


def imagen_a_data_url(imagen_pil):
    """Convierte una imagen PIL a base64 en formato PNG."""
    buffer = io.BytesIO()
    imagen_pil.save(buffer, format="PNG")
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


def preparar_imagen_usuario(pil_img):
    """Mantiene la imagen original en RGB y la procesa para el modelo MobileNetV2."""
    imagen = ImageOps.exif_transpose(pil_img).convert("RGB")
    tensor = tf.convert_to_tensor(np.asarray(imagen, dtype=np.float32))
    tensor = tf.image.resize(tensor, TARGET_SIZE)
    tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
    return tensor[tf.newaxis, ...].numpy()


def serializar_prediccion(probabilidades):
    probabilidades = np.asarray(probabilidades, dtype=np.float32)
    indices_ordenados = np.argsort(probabilidades)[::-1]

    top3 = [
        {
            "indice": int(indice),
            "clase": etiqueta_clase(CLASES[int(indice)]),
            "confianza": round(float(probabilidades[int(indice)]), 4),
        }
        for indice in indices_ordenados[:3]
    ]

    todas = [
        {
            "indice": i,
            "clase": etiqueta_clase(CLASES[i]),
            "confianza": round(float(probabilidades[i]), 4),
        }
        for i in range(len(CLASES))
    ]

    indice_max = int(np.argmax(probabilidades))
    prediccion = {
        "indice": indice_max,
        "clase": etiqueta_clase(CLASES[indice_max]),
        "confianza": round(float(probabilidades[indice_max]), 4),
    }

    return prediccion, top3, todas


def obtener_respuesta_modelo(imagen_entrada):
    if MODELO is None:
        return None, {"ok": False, "error": "El modelo no está entrenado. Ejecuta primero: python entrenar_moda.py"}, 503

    prediccion = MODELO.predict(imagen_entrada, verbose=0)[0]
    prediccion_obj, top3, todas = serializar_prediccion(prediccion)
    baja_confianza = bool(prediccion_obj["confianza"] < 0.60)

    resultado = {
        "ok": True,
        "prediccion": prediccion_obj,
        "top3": top3,
        "todas": todas,
        "baja_confianza": baja_confianza,
    }
    return prediccion_obj, resultado, 200


@app.route("/salud", methods=["GET"])
def salud():
    return jsonify({
        "ok": True,
        "modelo_cargado": MODELO is not None,
        "clases": [etiqueta_clase(clase) for clase in CLASES],
    })


@app.route("/muestras", methods=["GET"])
def obtener_muestras():
    muestras_dataset = obtener_muestras_galeria(request.args.get("cantidad", default=12, type=int) or 12)
    cantidad = max(1, min(len(muestras_dataset), len(muestras_dataset) if muestras_dataset else 1))

    muestras = []
    for item in muestras_dataset[:cantidad]:
        imagen = Image.open(item["ruta_imagen"]).convert("RGB")
        imagen = imagen.resize((224, 224), getattr(Image, "Resampling", Image).LANCZOS)
        muestras.append(
            {
                "id": int(item["id"]),
                "imagen": imagen_a_data_url(imagen),
                "clase_real": etiqueta_clase(item["articleType"]),
            }
        )

    return jsonify({"ok": True, "muestras": muestras})


@app.route("/predecir-muestra", methods=["POST"])
def predecir_muestra():
    if MODELO is None:
        return jsonify({"ok": False, "error": "El modelo no está entrenado. Ejecuta primero: python entrenar_moda.py"}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "id" not in data:
        return jsonify({"ok": False, "error": "Falta el campo 'id' en el cuerpo JSON."}), 400

    try:
        muestra_id = int(data["id"])
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "El campo 'id' debe ser un número entero."}), 400

    muestras_dataset = obtener_muestras_activas()
    item = next((entry for entry in muestras_dataset if int(entry["id"]) == muestra_id), None)
    if item is None:
        return jsonify({"ok": False, "error": "El identificador de la muestra no existe en el dataset real."}), 404

    imagen = Image.open(item["ruta_imagen"]).convert("RGB")
    entrada = preparar_imagen_usuario(imagen)
    _, resultado, status = obtener_respuesta_modelo(entrada)
    if status != 200:
        return jsonify(resultado), status

    imagen_preview = imagen.resize((224, 224), getattr(Image, "Resampling", Image).LANCZOS)
    resultado["clase_real"] = etiqueta_clase(item["articleType"])
    resultado["acierto"] = bool(resultado["prediccion"]["clase"] == resultado["clase_real"])
    resultado["imagen_original"] = imagen_a_data_url(imagen)
    resultado["preprocesada"] = imagen_a_data_url(imagen_preview)
    return jsonify(resultado), 200


@app.route("/predecir-imagen", methods=["POST"])
def predecir_imagen():
    if MODELO is None:
        return jsonify({"ok": False, "error": "El modelo no está entrenado. Ejecuta primero: python entrenar_moda.py"}), 503

    archivo = request.files.get("imagen")
    if archivo is None:
        return jsonify({"ok": False, "error": "Falta el archivo con el campo 'imagen'."}), 400

    nombre_archivo = (archivo.filename or "").lower()
    extension = Path(nombre_archivo).suffix
    validas = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    if extension not in validas:
        return jsonify({"ok": False, "error": "El archivo debe ser una imagen válida: PNG, JPG, JPEG, WEBP, BMP o GIF."}), 400

    if archivo.content_length and archivo.content_length > 5 * 1024 * 1024:
        return jsonify({"ok": False, "error": "La imagen supera los 5 MB permitidos."}), 400

    try:
        imagen = Image.open(archivo)
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        imagen_original = imagen.copy()

        entrada = preparar_imagen_usuario(imagen_original)
        _, resultado, status = obtener_respuesta_modelo(entrada)
        if status != 200:
            return jsonify(resultado), status

        imagen_redimensionada = imagen_original.resize((224, 224), getattr(Image, "Resampling", Image).LANCZOS)
        resultado["imagen_original"] = imagen_a_data_url(imagen_original)
        resultado["preprocesada"] = imagen_a_data_url(imagen_redimensionada)
        return jsonify(resultado), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": f"No se pudo procesar la imagen: {str(exc)}"}), 400


@app.route("/ejemplos/<int:indice_clase>", methods=["GET"])
def obtener_ejemplos(indice_clase):
    cantidad = request.args.get("cantidad", default=4, type=int)
    cantidad = max(1, min(cantidad, 8))

    if indice_clase < 0 or indice_clase >= len(CLASES):
        return jsonify({"ok": False, "error": "El índice de la clase está fuera del rango válido."}), 400

    clase = CLASES[indice_clase]
    ejemplos = []
    muestras_dataset = obtener_muestras_activas()
    for item in muestras_dataset:
        if item["articleType"] != clase:
            continue
        imagen = Image.open(item["ruta_imagen"]).convert("RGB")
        imagen = imagen.resize((224, 224), getattr(Image, "Resampling", Image).LANCZOS)
        ejemplos.append(imagen_a_data_url(imagen))
        if len(ejemplos) >= cantidad:
            break

    if not ejemplos:
        return jsonify({"ok": False, "error": "No hay ejemplos de entrenamiento para esta clase."}), 404

    return jsonify({"ok": True, "clase": clase, "ejemplos": ejemplos})


@app.errorhandler(404)
def error_404(error):
    return jsonify({"ok": False, "error": "La ruta solicitada no existe."}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
