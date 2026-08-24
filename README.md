# Laboratorio 4 - Análisis de Datos Geoespaciales

Este proyecto estudia la señal estimada de cianobacteria en los lagos Atitlán y
Amatitlán a partir de imágenes Sentinel-2. 

- **Parte 1** (ejercicios 1-8): serie temporal, mapas por fecha, persistencia
  espacial, correlaciones con NDVI y NDWI, comparación entre lagos y análisis
  exploratorio adicional.
- **Parte 2** (avance: ejercicios 1-3): construcción del dataset tabular para
  Machine Learning, variable respuesta binaria basada en el umbral de la OMS y
  selección de predictoras sin fuga de información.

El archivo principal es:

```text
notebooks/laboratorio-4-datos-geoespaciales.ipynb
```

El notebook está organizado para leerse de arriba hacia abajo. Las funciones más
largas viven en `src/`: `procesamiento_geoespacial.py` conserva el flujo de
openEO, `analisis_completo.py` ejecuta la consulta reproducible de las fechas
oficiales por medio de Sentinel-2 L2A en Planetary Computer y
`ml_cianobacteria.py` arma el conjunto de datos de la Parte 2.

## Estructura

```text
.
├── data/
│   ├── raw/                  insumos originales separados por lago
│   └── processed/            índices, tablas y figuras regenerables
├── notebooks/                cuaderno principal ejecutado
├── src/                      conexión, descarga y análisis
├── reports/                  informe final en PDF
├── codebook.md               datos, fechas, unidades y criterios
├── requirements.txt          dependencias de Python
└── README.md
```

## Cómo ejecutar el laboratorio

1. Crear un entorno e instalar las dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Abrir el notebook:

```bash
jupyter notebook notebooks/laboratorio-4-datos-geoespaciales.ipynb
```

3. Ejecutar de arriba hacia abajo. Si las tablas procesadas no están
   disponibles, el flujo consulta automáticamente las escenas oficiales y lee
   únicamente la ventana de cada lago.

4. Para repetir la descarga, cambiar `ACTUALIZAR_DATOS = True`. El proceso usa
   una resolución de análisis de 120 metros para mantener el laboratorio ligero.

## Salidas

- Tabla de métricas por lago y fecha.
- Mapas de cianobacteria para las 22 observaciones.
- Evolución temporal, extensión de valores altos y fechas críticas.
- Correlaciones de Cya con NDVI y NDWI.
- Mapas de persistencia, diferencias y distribuciones por fecha.
- Comparación entre lagos y lectura descriptiva por temporada.

Los datos procesados y las figuras regenerables no se versionan. El notebook
ejecutado conserva las salidas principales; el código, el informe y las
instrucciones sí quedan en el repositorio.

## Parte 2: dataset para Machine Learning

`src/ml_cianobacteria.py` convierte los resultados de la Parte 1 en una tabla
donde cada fila es un píxel de agua con coordenadas, fecha, lago, NDVI, NDWI y
Cya. Lee los cubos `data/processed/resultados/cubo_<lago>.npz` que genera
`python -m src.analisis_completo`, ya en EPSG:32615 y en metros.

```bash
python -m src.ml_cianobacteria   # autochequeo, no necesita datos
```

La respuesta usa el umbral de la OMS de 100 000 células/mL (100 en unidades
Se2WaQ), el mismo que la Parte 1 usó para los mapas.

Como `Cya = 115530.31 * ((B03 * B04) / B02) ** 2.38`, **B02, B03 y B04 quedan
excluidas como predictoras**. NDVI y NDWI comparten B04 y B03 con esa fórmula:
van en el dataset pero solo en el conjunto de predictoras "amplio", no en el
"estricto".

`src/modelos_cianobacteria.py` cubre el resto del laboratorio: Regresión
Logística, Random Forest y XGBoost con ajuste de hiperparámetros, validación
espacial por bloques de 1 km en EPSG:32615, generalización entre lagos, SHAP y
mapas predictivos.

```bash
python -m src.modelos_cianobacteria   # autochequeo, no necesita datos
```
