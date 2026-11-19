import { useEffect, useState } from "react";
import axios from "axios";
import { API_MODA } from "../config";
import "./ClasificadorPrendas.css";

const CLASES_POR_DEFECTO = [
  "Camisa",
  "Camiseta",
  "Pantalón",
  "Shorts",
  "Vestido",
  "Kurta",
  "Zapatos casuales",
  "Zapatillas deportivas",
  "Bolsos",
  "Accesorios",
];

export default function ClasificadorPrendas() {
  const [modeloActivo, setModeloActivo] = useState(true);
  const [mensajeServidor, setMensajeServidor] = useState("");
  const [muestras, setMuestras] = useState([]);
  const [muestraSeleccionada, setMuestraSeleccionada] = useState(null);
  const [resultado, setResultado] = useState(null);
  const [ejemplos, setEjemplos] = useState([]);
  const [mostrarClases, setMostrarClases] = useState(false);
  const [clasesModelo, setClasesModelo] = useState(CLASES_POR_DEFECTO);
  const [cargandoMuestras, setCargandoMuestras] = useState(false);
  const [cargandoResultado, setCargandoResultado] = useState(false);
  const [cargandoEjemplos, setCargandoEjemplos] = useState(false);
  const [archivo, setArchivo] = useState(null);
  const [vistaPrevia, setVistaPrevia] = useState("");
  const [dragActivo, setDragActivo] = useState(false);

  useEffect(() => {
    verificarServidor();
    cargarMuestras();
  }, []);

  useEffect(() => {
    return () => {
      if (vistaPrevia) {
        URL.revokeObjectURL(vistaPrevia);
      }
    };
  }, [vistaPrevia]);

  const verificarServidor = async () => {
    try {
      const respuesta = await axios.get(`${API_MODA}/salud`);
      const activo = respuesta.data && respuesta.data.modelo_cargado === true;
      const clasesServidor = Array.isArray(respuesta.data?.clases) && respuesta.data.clases.length > 0
        ? respuesta.data.clases
        : CLASES_POR_DEFECTO;

      setModeloActivo(activo);
      setClasesModelo(clasesServidor);
      setMensajeServidor(
        activo
          ? ""
          : "El servidor de prendas no está activo o el modelo no está entrenado. Ejecuta entrenar_moda.py y luego servidor_moda.py"
      );
    } catch (error) {
      setModeloActivo(false);
      setMensajeServidor(
        "El servidor de prendas no está activo o el modelo no está entrenado. Ejecuta entrenar_moda.py y luego servidor_moda.py"
      );
    }
  };

  const cargarMuestras = async () => {
    setCargandoMuestras(true);
    try {
      const respuesta = await axios.get(`${API_MODA}/muestras?cantidad=12`);
      setMuestras(respuesta.data.muestras || []);
    } catch (error) {
      setMensajeServidor("No se pudieron cargar las muestras del conjunto de prueba.");
    } finally {
      setCargandoMuestras(false);
    }
  };

  const cargarEjemplos = async (indiceClase) => {
    if (!Number.isInteger(indiceClase) || indiceClase < 0 || indiceClase >= clasesModelo.length) {
      return;
    }

    setCargandoEjemplos(true);
    try {
      const respuesta = await axios.get(`${API_MODA}/ejemplos/${indiceClase}?cantidad=4`);
      if (respuesta.data && respuesta.data.ok) {
        setEjemplos(respuesta.data.ejemplos || []);
      }
    } catch (error) {
      setEjemplos([]);
    } finally {
      setCargandoEjemplos(false);
    }
  };

  const validarArchivo = (archivoElegido) => {
    if (!archivoElegido) {
      return false;
    }

    if (!archivoElegido.type || !archivoElegido.type.startsWith("image/")) {
      setMensajeServidor("El archivo seleccionado no es una imagen válida.");
      return false;
    }

    if (archivoElegido.size > 5 * 1024 * 1024) {
      setMensajeServidor("La imagen supera el tamaño máximo permitido (5 MB).");
      return false;
    }

    return true;
  };

  const manejarArchivo = (archivoElegido) => {
    if (!validarArchivo(archivoElegido)) {
      return;
    }

    setArchivo(archivoElegido);
    setVistaPrevia(URL.createObjectURL(archivoElegido));
    setMensajeServidor("");
  };

  const manejarSeleccionMuestra = async (muestra) => {
    if (!modeloActivo) {
      return;
    }

    setMuestraSeleccionada(muestra.id);
    setResultado(null);
    setEjemplos([]);
    setCargandoResultado(true);
    setMostrarClases(false);

    try {
      const respuesta = await axios.post(`${API_MODA}/predecir-muestra`, { id: muestra.id });
      if (!respuesta.data.ok) {
        throw new Error(respuesta.data.error || "No se pudo clasificar la muestra.");
      }
      setResultado(respuesta.data);
      if (respuesta.data.prediccion && respuesta.data.prediccion.indice !== undefined) {
        await cargarEjemplos(respuesta.data.prediccion.indice);
      }
    } catch (error) {
      setMensajeServidor(error.response?.data?.error || error.message || "No se pudo procesar la muestra.");
    } finally {
      setCargandoResultado(false);
    }
  };

  const analizarImagen = async () => {
    if (!modeloActivo || !archivo) {
      return;
    }

    setCargandoResultado(true);
    setMensajeServidor("");
    setResultado(null);
    setEjemplos([]);

    const formData = new FormData();
    formData.append("imagen", archivo);

    try {
      const respuesta = await axios.post(`${API_MODA}/predecir-imagen`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (!respuesta.data.ok) {
        throw new Error(respuesta.data.error || "No se pudo analizar la imagen.");
      }

      setResultado(respuesta.data);
      if (respuesta.data.prediccion && respuesta.data.prediccion.indice !== undefined) {
        await cargarEjemplos(respuesta.data.prediccion.indice);
      }
    } catch (error) {
      setMensajeServidor(error.response?.data?.error || error.message || "Error al analizar la imagen.");
    } finally {
      setCargandoResultado(false);
    }
  };

  const limpiarTodo = () => {
    setResultado(null);
    setArchivo(null);
    setVistaPrevia("");
    setMuestraSeleccionada(null);
    setEjemplos([]);
    setMensajeServidor("");
    setMostrarClases(false);
  };

  return (
    <div className="prendas-panel">
      <div className="prendas-header">
        <h2>Reconocedor de Prendas con IA</h2>
      </div>

      {!modeloActivo && (
        <div className="prendas-aviso">
          El servidor de prendas no está activo o el modelo no está entrenado. Ejecuta entrenar_moda.py y luego servidor_moda.py
        </div>
      )}

      {mensajeServidor && <div className="prendas-error">{mensajeServidor}</div>}

      <div className="prendas-section">
        <h3>Elegir una imagen de ejemplo</h3>
        <div className="prendas-galeria-wrapper">
          {cargandoMuestras ? (
            <div className="prendas-spinner" />
          ) : (
            <div className="prendas-galeria">
              {muestras.map((muestra) => (
                <button
                  type="button"
                  key={muestra.id}
                  className={`prendas-muestra ${muestraSeleccionada === muestra.id ? "active" : ""}`}
                  onClick={() => manejarSeleccionMuestra(muestra)}
                >
                  <img src={muestra.imagen} alt={muestra.clase_real} />
                  <span>{muestra.clase_real}</span>
                </button>
              ))}
            </div>
          )}
          <button type="button" className="prendas-button secundario" onClick={cargarMuestras}>
            Cargar otras muestras
          </button>
        </div>
      </div>

      <div className="prendas-section">
        <h3>Subir tu propia imagen</h3>
        <div
          className={`prendas-dropzone ${dragActivo ? "active" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActivo(true);
          }}
          onDragLeave={() => setDragActivo(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragActivo(false);
            const archivoElegido = event.dataTransfer.files && event.dataTransfer.files[0];
            manejarArchivo(archivoElegido);
          }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={(event) => manejarArchivo(event.target.files && event.target.files[0])}
          />
          <span>Arrastra y suelta la imagen aquí o haz clic para elegir</span>
        </div>

        {vistaPrevia && (
          <div className="prendas-preview">
            <img src={vistaPrevia} alt="Vista previa de la prenda" />
          </div>
        )}

        <div className="prendas-actions">
          <button type="button" className="prendas-button primario" onClick={analizarImagen} disabled={!archivo || cargandoResultado || !modeloActivo}>
            {cargandoResultado ? "Analizando..." : "Analizar imagen"}
          </button>
          <button type="button" className="prendas-button secundario" onClick={limpiarTodo}>
            Limpiar
          </button>
        </div>
      </div>

      {cargandoResultado && <div className="prendas-spinner grande" />}

      {resultado && (
        <div className="prendas-resultados">
          <div className="prendas-resumen">
            <div>
              <span className="prendas-etiqueta">Prenda predicha</span>
              <h3>{resultado.prediccion.clase}</h3>
            </div>
            <div className="prendas-confianza">
              <span>{(resultado.prediccion.confianza * 100).toFixed(1)}%</span>
            </div>
          </div>

          {resultado.clase_real && (
            <p className="prendas-real">
              Clase real: <strong>{resultado.clase_real}</strong> · {resultado.acierto ? "Acertó" : "No acertó"}
            </p>
          )}

          <div className="prendas-top3">
            {resultado.top3.map((item) => (
              <div key={item.indice} className="prendas-top3-item">
                <div className="prendas-top3-header">
                  <span>{item.clase}</span>
                  <span>{(item.confianza * 100).toFixed(1)}%</span>
                </div>
                <div className="prendas-barra-bg">
                  <div className="prendas-barra" style={{ width: `${item.confianza * 100}%` }} />
                </div>
              </div>
            ))}
          </div>

          <button type="button" className="prendas-button secundario" onClick={() => setMostrarClases((estado) => !estado)}>
            {mostrarClases ? "Ocultar clases del modelo" : "Ver clases del modelo"}
          </button>

          {mostrarClases && (
            <div className="prendas-clases">
              {resultado.todas.map((item) => (
                <div key={item.indice} className="prendas-clase-row">
                  <div className="prendas-clase-meta">
                    <span>{item.clase}</span>
                    <span>{(item.confianza * 100).toFixed(1)}%</span>
                  </div>
                  <div className="prendas-barra-bg">
                    <div className="prendas-barra" style={{ width: `${item.confianza * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {resultado.preprocesada && (
            <div className="prendas-preprocesada">
              <h4>Imagen que ve el modelo</h4>
              <img src={resultado.preprocesada} alt="Preprocesado para la IA" />
            </div>
          )}

          {resultado.baja_confianza && (
            <div className="prendas-warning">
              La IA no está segura. Este modelo solo reconoce estas categorías: {clasesModelo.join(", ")}.
            </div>
          )}

          <div className="prendas-section ejemplos">
            <h3>Coincidencias con el entrenamiento</h3>
            {cargandoEjemplos ? (
              <div className="prendas-spinner" />
            ) : (
              <div className="prendas-ejemplos">
                {ejemplos.length > 0 ? (
                  ejemplos.map((ejemplo, index) => (
                    <img key={`${resultado.prediccion.indice}-${index}`} src={ejemplo} alt="Ejemplo de entrenamiento" />
                  ))
                ) : (
                  <p>No hay ejemplos disponibles para esta clase.</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
