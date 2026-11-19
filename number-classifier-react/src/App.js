import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { API_DIGITOS } from "./config";
import ClasificadorPrendas from "./components/ClasificadorPrendas";
import "./components/ClasificadorPrendas.css";
import "./App.css";

class Node {
  constructor(isNodeActive) {
    this.isNodeActive = isNodeActive;
  }
}

function CalculadoraNumeros() {
  const [numOfRows] = useState(28);
  const [numOfCols] = useState(28);

  const [grids, setGrids] = useState([[], [], []]);
  const [isUserDrawing, setIsUserDrawing] = useState([false, false, false]);
  const [resultNums, setResultNums] = useState([null, null, null]);
  const [emptyGrids, setEmptyGrids] = useState([true, true, true]);
  const [loadings, setLoadings] = useState([false, false, false]);

  const lastNodeRefs = [useRef(null), useRef(null), useRef(null)];

  const createAllGrids = useCallback(() => {
    const tempGrids = [[], [], []];
    for (let g = 0; g < 3; g++) {
      for (let i = 0; i < numOfCols; i++) {
        tempGrids[g][i] = [];
        for (let j = 0; j < numOfRows; j++) {
          tempGrids[g][i][j] = new Node(0);
        }
      }
    }
    setGrids(tempGrids);
    setEmptyGrids([true, true, true]);
    setResultNums([null, null, null]);
  }, [numOfCols, numOfRows]);

  useEffect(() => {
    createAllGrids();
  }, [createAllGrids]);

  const clearAllGrids = () => {
    const newGrids = [...grids];
    for (let g = 0; g < 3; g++) {
      for (let i = 0; i < numOfCols; i++) {
        for (let j = 0; j < numOfRows; j++) {
          newGrids[g][i][j] = new Node(0);
          const nodeElement = document.getElementById(`node-${g}-${i}-${j}`);
          if (nodeElement) nodeElement.className = "node-inactive";
        }
      }
    }
    setGrids(newGrids);
    setEmptyGrids([true, true, true]);
    setResultNums([null, null, null]);
  };

  const onMouseDownEvent = (gridIndex, i, j) => {
    const newDrawing = [...isUserDrawing];
    newDrawing[gridIndex] = true;
    setIsUserDrawing(newDrawing);

    if (emptyGrids[gridIndex]) {
      const newEmpty = [...emptyGrids];
      newEmpty[gridIndex] = false;
      setEmptyGrids(newEmpty);
    }

    pintarNodo(gridIndex, i, j);
    lastNodeRefs[gridIndex].current = { i, j };
  };

  const onMouseUpEvent = (gridIndex) => {
    const newDrawing = [...isUserDrawing];
    newDrawing[gridIndex] = false;
    setIsUserDrawing(newDrawing);
    lastNodeRefs[gridIndex].current = null;
  };

  const pintarNodo = (gridIndex, x, y) => {
    if (grids[gridIndex] && grids[gridIndex][x] && grids[gridIndex][x][y]) {
      const nodeElement = document.getElementById(`node-${gridIndex}-${x}-${y}`);
      const nodeReact = grids[gridIndex][x][y];

      nodeReact.isNodeActive = 255;
      if (nodeElement) {
        nodeElement.className = "node-active";
      }
    }
  };

  const drawLine = (gridIndex, x0, y0, x1, y1) => {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    let x = x0;
    let y = y0;

    while (true) {
      pintarNodo(gridIndex, x, y);
      if (x === x1 && y === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x += sx;
      }
      if (e2 < dx) {
        err += dx;
        y += sy;
      }
    }
  };

  const onMouseEnterEvent = (gridIndex, i, j) => {
    if (isUserDrawing[gridIndex]) {
      if (emptyGrids[gridIndex]) {
        const newEmpty = [...emptyGrids];
        newEmpty[gridIndex] = false;
        setEmptyGrids(newEmpty);
      }

      if (lastNodeRefs[gridIndex].current) {
        const { i: prevI, j: prevJ } = lastNodeRefs[gridIndex].current;
        drawLine(gridIndex, prevI, prevJ, i, j);
      } else {
        pintarNodo(gridIndex, i, j);
      }
      lastNodeRefs[gridIndex].current = { i, j };
    }
  };

  const sendAllNumbers = async () => {
    const newLoadings = [true, true, true];
    setLoadings(newLoadings);

    const promises = [0, 1, 2].map(async (gridIndex) => {
      if (emptyGrids[gridIndex]) return null;

      const numArr = [];
      for (let i = 0; i < numOfRows; i++) {
        for (let j = 0; j < numOfCols; j++) {
          numArr.push(grids[gridIndex][j][i].isNodeActive);
        }
      }

      try {
        const response = await axios.post(API_DIGITOS, numArr);
        return { gridIndex, data: response.data };
      } catch (error) {
        console.log(`Error en panel ${gridIndex + 1}:`, error);
        return { gridIndex, data: "Error" };
      }
    });

    const results = await Promise.all(promises);

    const newResults = [...resultNums];
    results.forEach((res) => {
      if (res) newResults[res.gridIndex] = res.data;
    });

    setResultNums(newResults);
    setLoadings([false, false, false]);
  };

  return (
    <div className="app-wrapper">
      <h1 className="main-title">Calculadora de Múltiples Números</h1>

      <div className="main-layout">
        <div className="panels-container">
          {[0, 1, 2].map((gridIndex) => (
            <div
              key={gridIndex}
              className="digit-panel"
              onMouseLeave={() => onMouseUpEvent(gridIndex)}
            >
              <div className="grid-container">
                <h2 className="grid-title">Panel {gridIndex + 1}</h2>
                <div className="grid-top-container">
                  {grids[gridIndex] &&
                    grids[gridIndex].map((row, i) => (
                      <div key={i} className="grid-row">
                        {row.map((col, j) => (
                          <div
                            key={`${gridIndex}-${i}-${j}`}
                            className={col.isNodeActive === 255 ? "node-active" : "node-inactive"}
                            id={`node-${gridIndex}-${i}-${j}`}
                            onMouseDown={() => onMouseDownEvent(gridIndex, i, j)}
                            onMouseUp={() => onMouseUpEvent(gridIndex)}
                            onMouseEnter={() => onMouseEnterEvent(gridIndex, i, j)}
                          />
                        ))}
                      </div>
                    ))}
                </div>
              </div>

              <div className="result-container">
                <h3 className="result-title">Predicción:</h3>
                <div className="server-response">
                  {loadings[gridIndex] ? <div className="loader-container-small" /> : resultNums[gridIndex]}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="global-sidebar-controls">
          <h2 className="sidebar-title">Controles</h2>
          <button className="send-button-global" onClick={sendAllNumbers}>
            ENVIAR
          </button>
          <button className="clear-button-global" onClick={clearAllGrids}>
            LIMPIAR
          </button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("numeros");

  return (
    <div className="app-shell">
      <div className="tab-bar">
        <button
          type="button"
          className={`tab-button ${activeTab === "numeros" ? "active" : ""}`}
          onClick={() => setActiveTab("numeros")}
        >
          Números
        </button>
        <button
          type="button"
          className={`tab-button ${activeTab === "prendas" ? "active" : ""}`}
          onClick={() => setActiveTab("prendas")}
        >
          Prendas
        </button>
      </div>

      <div className="tab-content">
        {activeTab === "numeros" ? <CalculadoraNumeros /> : <ClasificadorPrendas />}
      </div>
    </div>
  );
}
