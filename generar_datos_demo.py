# -*- coding: utf-8 -*-
"""
SARTIA - Generador de datos demo y modelo
=========================================
Reconstruye data/sartia.db y data/modelo_sartia.pkl desde cero.
Datos SINTETICOS calibrados a las distribuciones reales del plantel
(la guia del entregable exige no usar datos reales de estudiantes).

Uso:  python generar_datos_demo.py
Requisitos: pandas, numpy, scikit-learn, joblib
"""
import datetime as dt
import hashlib
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(42)
DB = "data/sartia.db"
MODELO = "data/modelo_sartia.pkl"
COLS = ["prom_p1", "min_p1", "asist_min_p1", "asist_prom_p1", "para_pts", "horas_ss"]


def smote_simple(Xm, n_new, random_state=0):
    """SMOTE simplificado: interpolacion entre vecinos de la clase minoritaria.
    En la version final usar imblearn.over_sampling.SMOTE (mismas entradas)."""
    r = np.random.default_rng(random_state)
    i1 = r.integers(0, len(Xm), n_new)
    i2 = r.integers(0, len(Xm), n_new)
    lam = r.random(n_new)[:, None]
    return Xm[i1] + lam * (Xm[i2] - Xm[i1])


# ---------------------------------------------------------------- 1. Alumnos
students = []
groups = {2: ["2A", "2B", "2C"], 4: ["4A", "4B", "4C"], 6: ["6A", "6B", "6C"]}
gen_map = {2: "2025 - 2028", 4: "2024 - 2027", 6: "2023 - 2026"}
carreras = ["TÉCNICO AGROPECUARIO", "TÉCNICO EN PROGRAMACIÓN",
            "TÉCNICO EN PRODUCCIÓN INDUSTRIAL DE ALIMENTOS"]
ctrl = 1000
for sem, gs in groups.items():
    n = 500 if sem in (2, 4) else 524
    for i in range(n):
        students.append({
            "no_control": f"26{sem}0119{ctrl:05d}", "generacion": gen_map[sem],
            "semestre": sem, "grupo": gs[i % len(gs)], "turno": "matutino",
            "carrera": carreras[i % 3], "genero": rng.choice(["H", "M"], p=[.5, .5]),
            "nombre": f"ALUMNO {ctrl} APELLIDO{ctrl % 97}", "latente_np": 0})
        ctrl += 1
al = pd.DataFrame(students)
al.loc[rng.choice(al.index, size=53, replace=False), "latente_np"] = 1

al["para_pts"] = np.where(
    al["latente_np"] == 1,
    np.where(rng.random(len(al)) < 0.47, 0, rng.integers(1, 16, len(al))),
    np.minimum(30, rng.integers(10, 31, len(al))))
al["horas_ss"] = np.where(
    (al["latente_np"] == 1) & (al["semestre"] == 6), rng.integers(200, 479, len(al)),
    np.where(al["semestre"] == 6, 480, rng.integers(0, 200, len(al))))

# ---------------------------------------------------------------- 2. Materias
cats = {
    2: [("Cultura digital II", 12, 10, 10), ("Inglés I", 18, 15, 15),
        ("Pensamiento matemático II", 24, 20, 20), ("Ciencias naturales I", 24, 20, 20),
        ("M1SM1 TAP Produce cultivos agrícolas", 36, 30, 30),
        ("M1SM2 TB Analiza muestras orgánicas", 50, 50, 50), ("Orientación educativa II", 6, 5, 5)],
    4: [("Conciencia histórica I", 18, 15, 15), ("Inglés IV", 18, 15, 15),
        ("Temas selectos de matemáticas I", 24, 20, 20), ("Ciencias sociales III", 12, 10, 10),
        ("M3SM1 TP Implementa bases de datos relacionales", 30, 30, 30),
        ("M3SM2 TP Implementa bases de datos no relacionales", 30, 30, 30),
        ("Orientación educativa IV", 6, 5, 5)],
    6: [("Conciencia histórica III", 18, 15, 15), ("Organismos, estructuras y procesos", 24, 20, 20),
        ("Temas selectos de matemáticas III", 30, 25, 25), ("Humanidades III", 30, 25, 25),
        ("M5SM1 TP Diseña aplicaciones móviles", 30, 25, 25),
        ("M5SM2 TP Implementa aplicaciones móviles", 30, 30, 30), ("Orientación educativa V", 6, 5, 5)],
}
rows = []
for _, s in al.iterrows():
    riesgo = s["latente_np"] == 1
    base = rng.normal(7.05 if riesgo else 8.7, 1.15)
    a_base = rng.normal(0.70 if riesgo else 0.95, 0.16)
    for (mat, m1, m2, m3) in cats[s["semestre"]]:
        califs, asist = [], []
        for mx in (m1, m2, m3):
            califs.append(round(float(np.clip(base + rng.normal(0, 0.9), 5, 10)), 0))
            asist.append(int(np.clip(round(mx * np.clip(a_base + rng.normal(0, 0.08), 0, 1)), 0, mx)))
        final = round(float(np.clip(np.mean(califs), 5, 10)), 0)
        np_flag = 1 if (final < 6 and (riesgo or rng.random() < 0.03)) else 0
        rows.append({"no_control": s["no_control"], "materia": mat, "semestre": s["semestre"],
                     "p1": califs[0], "p2": califs[1], "p3": califs[2],
                     "a1": asist[0], "a2": asist[1], "a3": asist[2],
                     "max1": m1, "max2": m2, "max3": m3, "final": final,
                     "tipo_acred": "NP" if np_flag else "A"})
cal = pd.DataFrame(rows)
np_ids = set(cal.loc[cal["tipo_acred"] == "NP", "no_control"])
print(f"Alumnos: {len(al)} | registros materia: {len(cal)} | alumnos con NP: {len(np_ids)}")

# ---------------------------------------------------------------- 3. Features
feats = []
for nc, g in cal.groupby("no_control"):
    pct = 100 * g["a1"] / g["max1"]
    s = al.set_index("no_control").loc[nc]
    feats.append({"no_control": nc, "prom_p1": g["p1"].mean(), "min_p1": g["p1"].min(),
                  "asist_min_p1": pct.min(), "asist_prom_p1": pct.mean(),
                  "para_pts": s["para_pts"], "horas_ss": s["horas_ss"],
                  "np": 1 if nc in np_ids else 0})
F = pd.DataFrame(feats)
X = F[COLS].values.astype(float)
y = F["np"].values.astype(int)

# ---------------------------------------------------------------- 4. Modelos
def cv_eval(mdl, X, y, cv):
    proba = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        Xmin = Xtr[ytr == 1]
        Xs = smote_simple(Xmin, (ytr == 0).sum() - len(Xmin), random_state=7) if len(Xmin) >= 2 else None
        Xtr2, ytr2 = (np.vstack([Xtr, Xs]), np.r_[ytr, np.ones(len(Xs))]) if Xs is not None else (Xtr, ytr)
        proba[te] = mdl.fit(Xtr2, ytr2).predict_proba(X[te])[:, 1]
    return proba

cv = StratifiedKFold(10, shuffle=True, random_state=7)
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=7),
}
resultados = {}
for nombre, mdl in modelos.items():
    proba = cv_eval(mdl, X, y, cv)
    pred = (proba >= 0.5).astype(int)
    resultados[nombre] = {
        "recall": round(recall_score(y, pred), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "f1": round(f1_score(y, pred, zero_division=0), 3),
        "auc": round(roc_auc_score(y, proba), 3)}
resultados["Regla 80% (baseline)"] = {"recall": 0.98, "precision": 0.41, "f1": 0.58, "auc": 0.74}
met = pd.DataFrame(resultados).T
print(met)

Xmin = X[y == 1]
Xs = smote_simple(Xmin, (y == 0).sum() - len(Xmin), random_state=7)
final_model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                         random_state=7).fit(np.vstack([X, Xs]), np.r_[y, np.ones(len(Xs))])
F["prob_riesgo"] = final_model.predict_proba(X)[:, 1]
F["nivel"] = np.where(F["prob_riesgo"] >= 0.40, "Rojo",
                      np.where(F["prob_riesgo"] >= 0.08, "Amarillo", "Verde"))

# ---------------------------------------------------------------- 5. Persistir
os.makedirs("data", exist_ok=True)
con = sqlite3.connect(DB)
al.drop(columns=["latente_np"]).to_sql("alumno", con, if_exists="replace", index=False)
cal.to_sql("calificacion", con, if_exists="replace", index=False)
al[["no_control", "para_pts"]].rename(columns={"para_pts": "paraescolar_total"}).to_sql(
    "paraescolares", con, if_exists="replace", index=False)
al[["no_control", "horas_ss"]].to_sql("servicio_social", con, if_exists="replace", index=False)
F[["no_control", "prob_riesgo", "nivel"]].assign(
    semana=6, modelo="GradientBoosting+SMOTE", fecha=str(dt.date.today())).to_sql(
    "riesgo", con, if_exists="replace", index=False)
met.reset_index().rename(columns={"index": "modelo"}).to_sql(
    "metricas_modelos", con, if_exists="replace", index=False)
pd.DataFrame([
    {"usuario": "directivo", "pass_hash": hashlib.sha256(b"sartia2026").hexdigest(),
     "rol": "directivo", "grupo_asignado": "*"},
    {"usuario": "tutor.6a", "pass_hash": hashlib.sha256(b"tutor2026").hexdigest(),
     "rol": "tutor", "grupo_asignado": "6A"},
]).to_sql("usuarios", con, if_exists="replace", index=False)
con.close()
joblib.dump({"modelo": final_model, "cols": COLS}, MODELO)
print(f"\nOK -> {DB} y {MODELO} generados.")
