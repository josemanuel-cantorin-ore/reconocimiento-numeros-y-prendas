from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "images_ia" / "myntradataset"
IMAGENES_DIR = BASE_DIR / "images_ia" / "images"
SALIDA_DIR = BASE_DIR / "salida_entrenamiento"

CLASES_ES = [
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
LABEL_TO_INDEX = {clase: indice for indice, clase in enumerate(CLASES_ES)}


def normalizar_clase(article_type):
    """Normaliza los tipos reales del dataset a una taxonomía balanceada y útil."""
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
    if any(token in texto for token in ["handbag", "handbags", "wallet", "wallets", "backpack", "backpacks"]):
        return "Handbags"
    if any(token in texto for token in ["watch", "watches", "sunglasses", "belt", "belts", "perfume", "socks"]):
        return "Accessories"
    return None


def localizar_styles():
    """Busca el archivo de metadatos del dataset real."""
    for nombre in ("styles.txt", "styles.csv"):
        ruta = DATASET_DIR / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError("No se encontró styles.csv ni styles.txt en images_ia/myntradataset/")


def cargar_dataset_balanceado():
    """Carga una taxonomía balanceada de 10 clases, con suficientes imágenes por clase."""
    style_path = localizar_styles()
    df = pd.read_csv(style_path, on_bad_lines="skip")
    df.columns = [str(col).strip() for col in df.columns]

    if "articleType" not in df.columns:
        raise ValueError("El CSV no contiene la columna 'articleType'.")

    df = df[["id", "articleType"]].copy()
    df["articleType"] = df["articleType"].map(normalizar_clase)
    df = df[df["articleType"].notna()].copy()
    df = df[df["articleType"].isin(CLASES_ES)].copy()
    df["ruta_imagen"] = df["id"].astype(str).map(lambda value: IMAGENES_DIR / f"{value}.jpg")
    df = df[df["ruta_imagen"].map(lambda ruta: ruta.exists())].copy()

    muestras = []
    for clase in CLASES_ES:
        categoria = df[df["articleType"] == clase].copy()
        if len(categoria) < 375:
            raise ValueError(
                f"La categoría '{clase}' tiene solo {len(categoria)} imágenes; se necesitan al menos 375 para entrenar de forma estable."
            )
        muestras.append(categoria.sample(n=375, random_state=42))

    dataset = pd.concat(muestras, ignore_index=True)
    dataset["label_index"] = dataset["articleType"].map(LABEL_TO_INDEX)
    return dataset.reset_index(drop=True)


def cargar_imagen(path, label):
    """Lee una imagen RGB y la prepara para MobileNetV2."""
    imagen = tf.io.read_file(path)
    imagen = tf.image.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, [224, 224])
    imagen = tf.keras.applications.mobilenet_v2.preprocess_input(imagen)
    return imagen, label


def construir_modelo():
    """Usa MobileNetV2 como base y reemplaza la cabeza por 5 clases."""
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False

    entradas = tf.keras.layers.Input(shape=(224, 224, 3), name="entrada")
    x = base_model(entradas, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    salidas = tf.keras.layers.Dense(len(CLASES_ES), activation="softmax", name="salidas")(x)
    return tf.keras.Model(entradas, salidas, name="modelo_ropa_rgb")


def main():
    """Entrena el modelo de clasificación con imágenes reales a color y clase separada para Shorts."""
    dataset = cargar_dataset_balanceado()

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    split = int(len(dataset) * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    val_df = dataset.iloc[val_idx].reset_index(drop=True)

    train_paths = train_df["ruta_imagen"].astype(str).to_numpy()
    train_labels = train_df["label_index"].astype(np.int32).to_numpy()
    val_paths = val_df["ruta_imagen"].astype(str).to_numpy()
    val_labels = val_df["label_index"].astype(np.int32).to_numpy()

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    modelo = construir_modelo()
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping],
        verbose=1,
    )

    val_loss, val_accuracy = modelo.evaluate(val_ds, verbose=0)
    print(f"Precisión final en validación: {val_accuracy:.4f}")

    SALIDA_DIR.mkdir(exist_ok=True)
    modelo.save(BASE_DIR / "modelo_moda.keras")

    with (BASE_DIR / "clases_moda.json").open("w", encoding="utf-8") as archivo:
        json.dump(CLASES_ES, archivo, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Pérdida por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.title("Precisión por época")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(SALIDA_DIR / "historial_entrenamiento.png", dpi=150)
    plt.close()

    print(f"Modelo guardado en: {BASE_DIR / 'modelo_moda.keras'}")
    print(f"Clases guardadas en: {BASE_DIR / 'clases_moda.json'}")
    print(f"Gráficos guardados en: {SALIDA_DIR / 'historial_entrenamiento.png'}")


if __name__ == "__main__":
    main()
