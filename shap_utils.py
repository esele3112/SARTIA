# -*- coding: utf-8 -*-
"""
SARTIA — Explicabilidad del modelo con SHAP.
============================================
Construye las features de los alumnos del ciclo ACTUAL (misma lógica que el
procesamiento del parcial en app.py) y calcula valores SHAP con TreeExplainer
sobre el GradientBoosting de producción.

Uso desde la app (Streamlit):
    feats = shap_utils.features_activos()          # DataFrame, una fila por alumno
    shap_df, base = shap_utils.explicar(pack, feats)   # contribuciones por alumno

`shap_df` tiene una columna por variable del modelo y una fila alineada con
`feats`; `base` es el valor esperado E[f(x)] (probabilidad base del modelo).
Valores SHAP positivos = empujan la probabilidad de riesgo hacia arriba.
"""
import sqlite3

import numpy as np
import pandas as pd

DB = "data/sartia.db"

# Nombres amigables de las variables (para gráficas y texto)
NOMBRES = {
    "prom_p1": "Promedio acumulado",
    "min_p1": "Calificación más baja",
    "asist_min_p1": "Asistencia mínima (%)",
    "asist_prom_p1": "Asistencia promedio (%)",
    "para_pts": "Puntos paraescolares",
    "horas_ss": "Horas de servicio social",
}

def features_activos(db_path: str = DB) -> pd.DataFrame:
    """Reconstruye el frame de features del modelo para los alumnos del ciclo
    en curso (los que tienen calificaciones con ciclo='actual').

    Replica la lógica del botón 'Procesar' de app.py: promedio y mínimo
    acumulados hasta el último parcial cargado, asistencia acumulada sobre
    (parcial × sesiones máximas), más paraescolares y servicio social.
    """
    con = sqlite3.connect(db_path)
    cal = pd.read_sql("SELECT * FROM calificacion WHERE ciclo='actual'", con)
    para = pd.read_sql("SELECT no_control, paraescolar_total FROM paraescolares", con)
    ss = pd.read_sql("SELECT no_control, horas_acumuladas FROM servicio_social", con)
    con.close()

    if cal.empty:
        return pd.DataFrame(columns=["no_control"])

    # Último parcial cargado: el mayor k con datos en alguna materia
    k = max(k for k in (1, 2, 3)
            if cal[f"p{k}"].notna().any() or cal[f"a{k}"].notna().any())

    P = cal[[f"p{i}" for i in range(1, k + 1)]]
    A = cal[[f"a{i}" for i in range(1, k + 1)]]
    cal = cal.copy()
    cal["prom"] = P.mean(axis=1)
    cal["minp"] = P.min(axis=1)
    cal["pct"] = (100 * A.sum(axis=1) / (k * cal["max1"])).clip(lower=0, upper=100)

    feats = cal.groupby("no_control").agg(
        prom_p1=("prom", "mean"), min_p1=("minp", "min"),
        asist_min_p1=("pct", "min"), asist_prom_p1=("pct", "mean")).reset_index()

    feats = feats.merge(para.rename(columns={"paraescolar_total": "para_pts"}),
                        on="no_control", how="left")
    feats = feats.merge(ss.rename(columns={"horas_acumuladas": "horas_ss"}),
                        on="no_control", how="left")

    # Misma imputación que app.py: media de la población (fallback fijo)
    feats["para_pts"] = feats["para_pts"].fillna(
        feats["para_pts"].mean() if feats["para_pts"].notna().any() else 15)
    feats["horas_ss"] = feats["horas_ss"].fillna(
        feats["horas_ss"].mean() if feats["horas_ss"].notna().any() else 240)
    return feats


def explicar(pack: dict, feats: pd.DataFrame):
    """Calcula los valores SHAP de todos los alumnos del frame.

    Devuelve (shap_df, base): shap_df alineado con feats (mismo orden de
    filas, index reseteado) con una columna por variable; base es el valor
    esperado del modelo (probabilidad base media).
    """
    import shap  # import local: carga numba solo cuando se usa

    cols = pack["cols"]
    X = feats[cols].to_numpy(dtype=float)
    explainer = shap.TreeExplainer(pack["modelo"])
    sv = explainer.shap_values(X)
    # Según el modelo/explicador puede venir (n, m) o lista/3D para binario
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    base = explainer.expected_value
    if np.ndim(base) > 0:
        base = float(np.asarray(base).ravel()[-1])
    shap_df = pd.DataFrame(sv, columns=cols)
    return shap_df, float(base)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def en_probabilidad(base: float, shap_df: pd.DataFrame):
    """Convierte las contribuciones (espacio log-odds) a puntos de probabilidad.

    Para cada fila y variable:  dp = p_total - sigmoide(base + S - s),
    con S la suma de SHAP de la fila. Devuelve (df_dp, p_total, p_base) con
    el aporte de cada variable en puntos de probabilidad (0-1).
    """
    S = shap_df.sum(axis=1).to_numpy()
    p_total = _sigmoid(base + S)
    p_base = float(_sigmoid(base))
    dp = pd.DataFrame(
        {c: p_total - _sigmoid(base + S - shap_df[c].to_numpy())
         for c in shap_df.columns},
        index=shap_df.index)
    return dp, p_total, p_base


def waterfall_alumno(pack: dict, row_feats: pd.Series, base: float,
                     row_shap: pd.Series):
    """Gráfica waterfall de SHAP para UN alumno, en puntos de probabilidad.

    Devuelve la figura matplotlib lista para st.pyplot (show=False).
    """
    import matplotlib.pyplot as plt
    import shap

    cols = pack["cols"]
    dp, p_total, p_base = en_probabilidad(
        base, row_shap.to_frame().T)
    exp = shap.Explanation(
        values=dp.iloc[0].to_numpy(),
        base_values=p_base,
        data=row_feats[cols].to_numpy(dtype=float),
        feature_names=[NOMBRES.get(c, c) for c in cols])
    plt.close("all")
    shap.plots.waterfall(exp, max_display=7, show=False)
    return plt.gcf()


# Formato del valor observado por variable (para el texto)
VALOR_TEXTO = {
    "prom_p1": lambda v: f"promedio de {v:.1f}",
    "min_p1": lambda v: f"mínima de {v:.1f}",
    "asist_min_p1": lambda v: f"asistencia mínima de {v:.0f} %",
    "asist_prom_p1": lambda v: f"asistencia promedio de {v:.0f} %",
    "para_pts": lambda v: f"{v:.0f} pts paraescolares",
    "horas_ss": lambda v: f"{v:.0f} h de servicio social",
}


def drivers_alumno(row_feats: pd.Series, row_shap: pd.Series, top: int = 3):
    """Traduce las contribuciones SHAP de UN alumno a lenguaje natural.

    Devuelve (suben, bajan): listas como
    ['Promedio acumulado (promedio de 6.4)', ...] ordenadas por impacto.
    La dirección (sube/baja el riesgo) sale del signo exacto del valor SHAP;
    la magnitud se aprecia en la gráfica waterfall.
    """
    orden = row_shap.sort_values(ascending=False)
    suben, bajan = [], []
    for var in orden.index:
        nombre = NOMBRES.get(var, var)
        val = row_shap[var]
        texto_val = VALOR_TEXTO.get(var, lambda v: f"{v:.1f}")(row_feats[var])
        if val > 0.01 and len(suben) < top:
            suben.append(f"{nombre} ({texto_val})")
        elif val < -0.01 and len(bajan) < top:
            bajan.append(f"{nombre} ({texto_val})")
    return suben, bajan
