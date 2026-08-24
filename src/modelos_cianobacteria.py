"""Laboratorio 4, Parte 2. Modelos, validacion espacial y explicabilidad.

Ejercicios 4 a 9: construccion y evaluacion de modelos, validacion espacial con
bloques de 1 km, generalizacion entre lagos, SHAP y mapas predictivos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from src.procesamiento_geoespacial import CARPETA_FIGURAS

TAMANIO_BLOQUE_M = 1000.0
SEMILLA = 42


def dividir_train_test(datos: pd.DataFrame, X: List[str], y: str = "alta_presencia"):
    """4.1: division 70/30 estratificada por la clase."""

    return train_test_split(
        datos[X], datos[y], test_size=0.30, random_state=SEMILLA,
        stratify=datos[y],
    )


REJILLAS = {
    "Regresion Logistica": {
        "modelo": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "parametros": {"C": [0.01, 0.1, 1, 10]},
    },
    "Random Forest": {
        "modelo": RandomForestClassifier(class_weight="balanced", random_state=SEMILLA, n_jobs=-1),
        "parametros": {"n_estimators": [200, 400], "max_depth": [6, 12, None]},
    },
    "XGBoost": {
        "modelo": None,  # se arma en entrenar_modelos, necesita scale_pos_weight
        "parametros": {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
    },
}


def entrenar_modelos(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """4.2 y 4.3: ajusta hiperparametros con GridSearchCV, cv=3, scoring F1.

    F1 en vez de accuracy porque la clase positiva es minoritaria y el error
    caro (falso negativo) pesa en ambos terminos de F1.
    """

    peso_positivo = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    modelos = {}
    for nombre, config in REJILLAS.items():
        base = config["modelo"]
        if nombre == "XGBoost":
            base = XGBClassifier(
                eval_metric="logloss", random_state=SEMILLA,
                scale_pos_weight=peso_positivo, n_jobs=-1,
            )
        busqueda = GridSearchCV(base, config["parametros"], scoring="f1", cv=3, n_jobs=-1)
        busqueda.fit(X_train, y_train)
        modelos[nombre] = busqueda.best_estimator_
        print(f"{nombre}: mejores parametros {busqueda.best_params_}")
    return modelos


def evaluar_modelos(modelos: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series):
    """5.1: metricas minimas y matriz de confusion por modelo."""

    filas, matrices = [], {}
    for nombre, modelo in modelos.items():
        pred = modelo.predict(X_test)
        prob = modelo.predict_proba(X_test)[:, 1]
        filas.append({
            "modelo": nombre,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        })
        matrices[nombre] = confusion_matrix(y_test, pred)
    return pd.DataFrame(filas).set_index("modelo").round(3), matrices


def graficar_matrices_confusion(matrices: Dict[str, np.ndarray], sufijo: str = "") -> Path:
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.6), dpi=140)
    for eje, (nombre, matriz) in zip(np.atleast_1d(ejes), matrices.items()):
        eje.imshow(matriz, cmap="Blues")
        for (i, j), v in np.ndenumerate(matriz):
            eje.text(j, i, str(v), ha="center", va="center")
        eje.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        eje.set_yticks([0, 1], ["Real 0", "Real 1"])
        eje.set_title(nombre, fontsize=10)
    fig.tight_layout()
    carpeta = CARPETA_FIGURAS / "ml"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"matrices_confusion{sufijo}.png"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def asignar_bloques(datos: pd.DataFrame, tamanio_m: float = TAMANIO_BLOQUE_M) -> pd.DataFrame:
    """6.1: cuadricula regular en EPSG:32615 y asignacion de bloque por pixel."""

    salida = datos.copy()
    ix = np.floor(salida["x"] / tamanio_m).astype(int)
    iy = np.floor(salida["y"] / tamanio_m).astype(int)
    salida["bloque"] = salida["lago"] + "_" + ix.astype(str) + "_" + iy.astype(str)
    return salida


def resumen_bloques(datos: pd.DataFrame) -> pd.DataFrame:
    """6.1: numero de bloques y observaciones por bloque, por lago."""

    conteo = datos.groupby(["lago", "bloque"]).size().rename("observaciones")
    return conteo.groupby("lago").agg(
        n_bloques=("count"), obs_promedio=("mean"), obs_min=("min"), obs_max=("max")
    )


def graficar_bloques(datos: pd.DataFrame) -> Path:
    """6.2: mapa de los bloques espaciales generados."""

    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=140)
    for eje, lago in zip(ejes, sorted(datos["lago"].unique())):
        sub = datos[datos["lago"] == lago]
        codigos = sub["bloque"].astype("category").cat.codes
        eje.scatter(sub["x"], sub["y"], c=codigos, s=2, cmap="tab20")
        eje.set_title(f"{lago}: {sub['bloque'].nunique()} bloques")
        eje.set_xlabel("x (m, EPSG:32615)")
        eje.set_ylabel("y (m)")
    fig.tight_layout()
    carpeta = CARPETA_FIGURAS / "ml"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "bloques_espaciales.png"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def validacion_espacial(datos: pd.DataFrame, X: List[str], y: str = "alta_presencia", n_splits: int = 5):
    """6.3 y 6.4: GroupKFold por bloque espacial, mismos modelos que en 4."""

    n_grupos = datos["bloque"].nunique()
    n_splits = min(n_splits, n_grupos)
    grupos = GroupKFold(n_splits=n_splits)

    resultados = {nombre: [] for nombre in REJILLAS}
    peso_positivo = (datos[y] == 0).sum() / max((datos[y] == 1).sum(), 1)
    for entrena_idx, prueba_idx in grupos.split(datos[X], datos[y], datos["bloque"]):
        X_tr, X_te = datos[X].iloc[entrena_idx], datos[X].iloc[prueba_idx]
        y_tr, y_te = datos[y].iloc[entrena_idx], datos[y].iloc[prueba_idx]

        modelos = {
            "Regresion Logistica": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEMILLA, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=6, eval_metric="logloss",
                random_state=SEMILLA, scale_pos_weight=peso_positivo, n_jobs=-1,
            ),
        }
        for nombre, modelo in modelos.items():
            modelo.fit(X_tr, y_tr)
            prob = modelo.predict_proba(X_te)[:, 1]
            pred = modelo.predict(X_te)
            resultados[nombre].append({
                "f1": f1_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_te, prob) if y_te.nunique() > 1 else np.nan,
            })

    filas = [
        {"modelo": nombre, **pd.DataFrame(metricas).mean().to_dict()}
        for nombre, metricas in resultados.items()
    ]
    return pd.DataFrame(filas).set_index("modelo").round(3)


def comparar_aleatoria_vs_espacial(aleatoria: pd.DataFrame, espacial: pd.DataFrame) -> pd.DataFrame:
    """6.5: diferencia de desempeno entre validacion aleatoria y espacial."""

    comparacion = aleatoria[["f1", "roc_auc"]].join(
        espacial[["f1", "roc_auc"]], lsuffix="_aleatoria", rsuffix="_espacial"
    )
    comparacion["diferencia_f1"] = (
        comparacion["f1_espacial"] - comparacion["f1_aleatoria"]
    ).round(3)
    comparacion["diferencia_roc_auc"] = (
        comparacion["roc_auc_espacial"] - comparacion["roc_auc_aleatoria"]
    ).round(3)
    return comparacion.round(3)


def generalizacion_entre_lagos(datos: pd.DataFrame, X: List[str], y: str = "alta_presencia"):
    """7: entrena en un lago, evalua en el otro, y viceversa."""

    resultados = {}
    for entrena, evalua in (("Atitlan", "Amatitlan"), ("Amatitlan", "Atitlan")):
        train = datos[datos["lago"] == entrena]
        test = datos[datos["lago"] == evalua]
        peso_positivo = (train[y] == 0).sum() / max((train[y] == 1).sum(), 1)
        modelo = XGBClassifier(
            n_estimators=300, max_depth=6, eval_metric="logloss",
            random_state=SEMILLA, scale_pos_weight=peso_positivo, n_jobs=-1,
        )
        modelo.fit(train[X], train[y])
        prob = modelo.predict_proba(test[X])[:, 1]
        pred = modelo.predict(test[X])
        resultados[f"entrena={entrena} / evalua={evalua}"] = {
            "n_entrenamiento": len(train),
            "n_evaluacion": len(test),
            "f1": f1_score(test[y], pred, zero_division=0),
            "recall": recall_score(test[y], pred, zero_division=0),
            "roc_auc": roc_auc_score(test[y], prob) if test[y].nunique() > 1 else np.nan,
        }
    return pd.DataFrame(resultados).T.round(3)


def importancia_variables(modelo, X: List[str]) -> pd.DataFrame:
    """8.1: importancia global de las predictoras del mejor modelo."""

    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
    else:
        valores = np.abs(modelo.coef_[0])
    return (
        pd.DataFrame({"variable": X, "importancia": valores})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def graficar_importancia(tabla: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(6, 4), dpi=140)
    eje.barh(tabla["variable"][::-1], tabla["importancia"][::-1], color="#2a9d8f")
    eje.set_xlabel("Importancia")
    eje.set_title("Importancia global de las predictoras")
    fig.tight_layout()
    carpeta = CARPETA_FIGURAS / "ml"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "importancia_variables.png"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def resumen_shap(modelo, X_muestra: pd.DataFrame, guardar: bool = True):
    """8.2 y 8.3: SHAP summary plot e influencia por variable."""

    import shap
    import matplotlib.pyplot as plt

    explicador = shap.TreeExplainer(modelo)
    valores = explicador.shap_values(X_muestra)

    if guardar:
        carpeta = CARPETA_FIGURAS / "ml"
        carpeta.mkdir(parents=True, exist_ok=True)
        plt.figure(dpi=140)
        shap.summary_plot(valores, X_muestra, show=False)
        plt.tight_layout()
        plt.savefig(carpeta / "shap_summary.png")
        plt.close()

    influencia = pd.DataFrame({
        "variable": X_muestra.columns,
        "shap_medio_abs": np.abs(valores).mean(axis=0),
        "correlacion_valor_shap": [
            np.corrcoef(X_muestra[c], valores[:, i])[0, 1]
            for i, c in enumerate(X_muestra.columns)
        ],
    }).sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
    return influencia


def mapa_predictivo(modelo, datos: pd.DataFrame, X: List[str], nombre_lago: str) -> Path:
    """9: probabilidad de alta presencia reconstruida espacialmente, promedio
    de todas las fechas disponibles del lago."""

    import matplotlib.pyplot as plt

    sub = datos[datos["lago"] == nombre_lago].copy()
    sub["probabilidad"] = modelo.predict_proba(sub[X])[:, 1]
    promedio = sub.groupby(["x", "y"], as_index=False)["probabilidad"].mean()

    fig, eje = plt.subplots(figsize=(6, 5), dpi=140)
    puntos = eje.scatter(
        promedio["x"], promedio["y"], c=promedio["probabilidad"],
        cmap="RdYlGn_r", s=4, vmin=0, vmax=1,
    )
    fig.colorbar(puntos, ax=eje, label="Probabilidad de alta presencia")
    eje.set_title(f"Mapa predictivo — {nombre_lago}\n(muy baja < 0.25 ≤ baja < 0.5 ≤ alta < 0.75 ≤ muy alta)")
    eje.set_xlabel("x (m)")
    eje.set_ylabel("y (m)")
    fig.tight_layout()
    carpeta = CARPETA_FIGURAS / "ml"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"mapa_predictivo_{nombre_lago.lower()}.png"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def analisis_errores(modelo, datos: pd.DataFrame, X: List[str], nombre_lago: str, umbral: float = 0.5) -> Dict[str, float]:
    """9: falsos positivos y negativos, agregados espacialmente por lago."""

    sub = datos[datos["lago"] == nombre_lago].copy()
    sub["probabilidad"] = modelo.predict_proba(sub[X])[:, 1]
    sub["prediccion"] = (sub["probabilidad"] >= umbral).astype(int)
    falsos_positivos = ((sub["prediccion"] == 1) & (sub["alta_presencia"] == 0)).mean()
    falsos_negativos = ((sub["prediccion"] == 0) & (sub["alta_presencia"] == 1)).mean()
    aciertos = (sub["prediccion"] == sub["alta_presencia"]).mean()
    return {
        "lago": nombre_lago,
        "pct_aciertos": round(aciertos * 100, 2),
        "pct_falsos_positivos": round(falsos_positivos * 100, 2),
        "pct_falsos_negativos": round(falsos_negativos * 100, 2),
    }


def demo() -> None:
    """Chequeo minimo con datos sinteticos, no necesita el dataset real."""

    n = 600
    generador = np.random.default_rng(0)
    lago = np.where(generador.random(n) < 0.5, "Atitlan", "Amatitlan")
    x = generador.uniform(680_000, 700_000, n)
    y = generador.uniform(1_600_000, 1_620_000, n)
    b08 = generador.uniform(0, 0.2, n)
    respuesta = (b08 > 0.12).astype(int)  # senal fuerte y facil de aprender
    datos = pd.DataFrame({
        "lago": lago, "x": x, "y": y, "B08": b08,
        "mes": generador.integers(1, 13, n),
        "dia_del_anio": generador.integers(1, 366, n),
        "epoca_lluviosa": generador.integers(0, 2, n),
        "es_amatitlan": (lago == "Amatitlan").astype(int),
        "dist_centro": generador.uniform(0, 5000, n),
        "alta_presencia": respuesta,
    })

    X = ["B08", "mes", "dia_del_anio", "epoca_lluviosa", "es_amatitlan", "x", "y", "dist_centro"]
    X_tr, X_te, y_tr, y_te = dividir_train_test(datos, X)
    assert len(X_tr) + len(X_te) == n

    modelos = {
        "Regresion Logistica": LogisticRegression(max_iter=500, class_weight="balanced").fit(X_tr, y_tr),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr),
    }
    metricas, matrices = evaluar_modelos(modelos, X_te, y_te)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(metricas.columns)
    assert metricas["f1"].max() > 0.5, "el modelo no aprendio ni la senal facil"

    con_bloques = asignar_bloques(datos, tamanio_m=1000)
    assert con_bloques["bloque"].nunique() > 1
    resumen = resumen_bloques(con_bloques)
    assert (resumen["n_bloques"] > 0).all()

    espacial = validacion_espacial(con_bloques, X, n_splits=3)
    assert set(espacial.index) == set(REJILLAS)

    comparacion = comparar_aleatoria_vs_espacial(
        metricas.reindex(espacial.index)[["f1", "roc_auc"]].assign(**{"f1": metricas["f1"], "roc_auc": metricas["roc_auc"]}),
        espacial,
    )
    assert "diferencia_f1" in comparacion.columns

    generalizacion = generalizacion_entre_lagos(datos, X)
    assert len(generalizacion) == 2

    importancia = importancia_variables(modelos["Random Forest"], X)
    assert set(importancia["variable"]) == set(X)

    errores = analisis_errores(modelos["Random Forest"], datos.assign(alta_presencia=respuesta), X, "Atitlan")
    assert 0 <= errores["pct_aciertos"] <= 100

    print("demo ok")


if __name__ == "__main__":
    demo()
