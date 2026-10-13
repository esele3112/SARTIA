# -*- coding: utf-8 -*-
"""
SARTIA — Entrenamiento del modelo con datos REALES del ciclo anterior.
========================================================================
Lee data/sartia.db (generada por etl_sartia.py con el detalle de
calificaciones completo: parciales 1-3 + calificación final), construye las
features acumuladas al parcial 1 y entrena el GradientBoosting + SMOTE que
usa el dashboard para predecir durante el ciclo en curso.

Etiqueta (y): asignatura NO acreditada al final del ciclo
             (tipo_acred en {'NA','NAC'} o final < 6).

Salida:
  - data/modelo_sartia.pkl   {"modelo": <pipeline>, "cols": COLS}
  - actualiza la tabla metricas_modelos de la base con las métricas reales.

Uso:  python entrenar_modelo.py
Requisitos: pandas, numpy, scikit-learn, joblib (imblearn opcional).
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

DB = "data/sartia.db"
MODELO = "data/modelo_sartia.pkl"
COLS = ["prom_p1", "min_p1", "asist_min_p1", "asist_prom_p1", "para_pts", "horas_ss"]
UMBRAL_REPROBATORIA = 6.0
UMBRAL_CLASE = 0.5          # corte de probabilidad para métricas (el dashboard usa 0.40/0.08 para el semáforo)


def smote(Xm, n_new, random_state=0):
    """SMOTE: imblearn si está instalado, si no interpolación simple entre
    vecinos de la clase minoritaria (equiv. al demo)."""
    try:
        from imblearn.over_sampling import SMOTE
        Xs, _ = SMOTE(random_state=random_state, k_neighbors=max(1, min(5, len(Xm) - 1))
                      ).fit_resample(np.vstack([Xm, Xm]), np.r_[np.ones(len(Xm)), np.ones(len(Xm))])
        # fit_resample duplica; tomamos solo las sintéticas generadas
        return Xs[len(Xm):][:n_new] if len(Xs) > len(Xm) else Xs[:n_new]
    except Exception:
        r = np.random.default_rng(random_state)
        if len(Xm) < 2:
            return np.repeat(Xm, max(1, n_new // max(1, len(Xm))), axis=0)[:n_new]
        i1 = r.integers(0, len(Xm), n_new)
        i2 = r.integers(0, len(Xm), n_new)
        lam = r.random(n_new)[:, None]
        return Xm[i1] + lam * (Xm[i2] - Xm[i1])


def construir_features(con):
    """Features acumuladas al parcial 1 + etiqueta de la final del ciclo."""
    cal = pd.read_sql("SELECT * FROM calificacion", con)
    para = pd.read_sql("SELECT no_control, paraescolar_total FROM paraescolares", con)
    ss = pd.read_sql("SELECT no_control, horas_acumuladas FROM servicio_social", con)

    pct = 100 * cal["a1"] / cal["max1"]
    feats = cal.assign(asist_pct=pct).groupby("no_control").agg(
        prom_p1=("p1", "mean"), min_p1=("p1", "min"),
        asist_min_p1=("asist_pct", "min"), asist_prom_p1=("asist_pct", "mean")).reset_index()

    # Etiqueta: ¿reprobó alguna materia al final del ciclo?
    cal["no_acreditada"] = (
        cal["tipo_acred"].astype(str).str.strip().str.upper().isin(["NA", "NAC"]) |
        (pd.to_numeric(cal["final"], errors="coerce") < UMBRAL_REPROBATORIA).fillna(False))
    y = cal.groupby("no_control")["no_acreditada"].max().astype(int).rename("np")

    F = feats.merge(y, on="no_control", how="inner")
    F = F.merge(para.rename(columns={"paraescolar_total": "para_pts"}), on="no_control", how="left")
    F = F.merge(ss.rename(columns={"horas_acumuladas": "horas_ss"}), on="no_control", how="left")
    F["para_pts"] = F["para_pts"].fillna(F["para_pts"].mean()).fillna(0)
    F["horas_ss"] = F["horas_ss"].fillna(F["horas_ss"].mean()).fillna(0)
    return F


def cv_eval(mdl, X, y, cv):
    proba = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        Xmin = Xtr[ytr == 1]
        n_min = len(Xmin)
        n_new = int((ytr == 0).sum()) - n_min
        if n_min >= 2 and n_new > 0:
            Xs = smote(Xmin, n_new, random_state=7)
            Xtr = np.vstack([Xtr, Xs])
            ytr = np.r_[ytr, np.ones(len(Xs))]
        proba[te] = mdl.fit(Xtr, ytr).predict_proba(X[te])[:, 1]
    return proba


def main():
    con = sqlite3.connect(DB)
    F = construir_features(con)
    con.close()

    F = F.dropna(subset=COLS + ["np"])
    X = F[COLS].values.astype(float)
    y = F["np"].values.astype(int)
    print(f"Alumnos con etiqueta: {len(F)}  |  positivos (reprobaron): {int(y.sum())} "
          f"({100 * y.mean():.1f} %)")

    cv = StratifiedKFold(10, shuffle=True, random_state=7)
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=500),
        "Gradient Boosting + SMOTE (producción)": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=7),
    }
    filas = []
    for nombre, mdl in modelos.items():
        proba = cv_eval(mdl, X, y, cv)
        pred = (proba >= UMBRAL_CLASE).astype(int)
        filas.append({
            "modelo": nombre,
            "recall": round(recall_score(y, pred, zero_division=0), 3),
            "precision": round(precision_score(y, pred, zero_division=0), 3),
            "f1": round(f1_score(y, pred, zero_division=0), 3),
            "auc_roc": round(roc_auc_score(y, proba), 3)})
        print(filas[-1])

    # Modelo final: entrenado con TODOS los datos + SMOTE
    Xmin = X[y == 1]
    n_new = int((y == 0).sum()) - len(Xmin)
    if len(Xmin) >= 2 and n_new > 0:
        Xs = smote(Xmin, n_new, random_state=7)
        Xf, yf = np.vstack([X, Xs]), np.r_[y, np.ones(len(Xs))]
    else:
        Xf, yf = X, y
    final_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=7).fit(Xf, yf)

    joblib.dump({"modelo": final_model, "cols": COLS}, MODELO)

    # Refrescar métricas reales en la base (las ve la pantalla P06)
    met = pd.DataFrame(filas)
    con = sqlite3.connect(DB)
    met.to_sql("metricas_modelos", con, if_exists="replace", index=False)
    con.close()
    print(f"\nOK -> {MODELO} entrenado con datos reales y metricas_modelos actualizada.")


if __name__ == "__main__":
    main()
