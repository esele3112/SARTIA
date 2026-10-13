# -*- coding: utf-8 -*-
"""
SARTIA — Sistema de Alerta Temprana de Riesgo Académico (CBTA 197)
MVP funcional · Streamlit + SQLite + scikit-learn
Ejecutar:  streamlit run app.py
Cuentas de prueba:  directivo/sartia2026 · tutor.6a/tutor2026

Flujo del ciclo escolar:
  1. Al inicio del ciclo: ETL con los archivos del ciclo ANTERIOR (con
     calificaciones finales) y entrenamiento del modelo (entrenar_modelo.py).
  2. Semana ~6:  Cargar parcial 1 -> predicciones -> semáforo actualizado.
  3. Semana ~11: Cargar parcial 2 -> predicciones actualizadas.
  4. Semana ~16: Cargar parcial 3 -> predicciones actualizadas.
  5. Al cerrar el ciclo: repetir el paso 1 con los archivos completos.
"""
import datetime as dt
import hashlib
import os
import secrets
import sqlite3
import string
import unicodedata
import re

def _nueva_password(n: int = 10) -> str:
    """Contraseña aleatoria única (letras + dígitos)."""
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SARTIA · CBTA 197", page_icon="⚠", layout="wide")

DB = "data/sartia.db"
MODEL = "data/modelo_sartia.pkl"

# Semana del ciclo en que se captura cada parcial (para el registro en riesgo)
SEMANA_POR_PARCIAL = {1: 6, 2: 11, 3: 16}

# Catálogo de sesiones máximas por materia (parcial 1) — transcrito del PDF del plantel
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", s.upper())

CATALOGO_P1 = {  # nombre normalizado -> máximo de sesiones del parcial 1
    "CONCIENCIAHISTORICAI": 18, "CONCIENCIAHISTORICAIII": 18, "INGLESI": 18, "INGLESIV": 18,
    "CULTURADIGITALII": 12, "PENSAMIENTOMATEMATICOII": 24, "TEMASSELECTOSDEMATEMATICASI": 24,
    "TEMASSELECTOSDEMATEMATICASIII": 30, "CIENCIASNATURALESI": 24, "CIENCIASSOCIALESII": 12,
    "CIENCIASSOCIALESIII": 12, "HUMANIDADESIII": 30, "ORGANISMOS": 24, "ORIENTACIONEDUCATIVA": 6,
    "FORMACIONSOCIOEMOCIONAL": 6, "APLICACIONESDELAADMINISTRACION": 18, "ECONOMIAPUBLICAYSOCIAL": 18,
    "PRODUCECULTIVOS": 102, "ANALIZAMUESTRAS": 50, "GESTIONAPROCESOS": 50, "VALIDALAMATERIA": 55,
    "DESARROLLASOFTWARE": 85, "DISEÑASOFTWARE": 30, "CODIFICASOFTWARE": 30, "IMPLEMENTASOFTWARE": 25,
    "PROCESAPRODUCTOSAGROPECUARIOS": 90, "APLICATECNICASDEPROPAGACION": 60, "EJECUTAPLANDENEGOCIOS": 80,
    "PROCESACARNE": 60, "ADMINISTRABASESDEDATOS": 60, "IMPLEMENTABASEDEDATOSRELACIONALES": 30,
    "DESARROLLAMODELODENEGOCIOS": 54, "APOYAENLAIDENTIFICACION": 24, "PARTICIPAENLAELABORACION": 30,
    "ASISTEENELDESARROLLO": 30, "REALIZALOSANALISIS": 30, "PROCESAALIMENTOS": 60,
    "DESARROLLAAPLICACIONESMOVILES": 60, "REALIZAANALISISFISICOS": 30, "TRANSFORMACARNE": 30,
    "REALIZAPROCESOSDETRANSFORMACION": 30, "DISENAPLAN": 24, "DISENAAPLICACIONES": 30,
}
def max_p1(materia):
    k = norm(materia)
    for clave, mx in CATALOGO_P1.items():
        if clave in k or k in clave:
            return mx
    return 24  # valor típico si la materia no está en el catálogo

# Siglas de especialidad detectadas por fragmento distintivo del nombre de la
# carrera (la columna CARRERA del sistema escolar viene con codificación
# dañada, por eso no se busca el nombre completo). ORDEN IMPORTA: TAE antes
# que TA, porque "...Emprendimiento AGROPECUARIO" contiene "AGROPECUAR...".
SIGLAS_PATRON = [
    ("PROGRAMACI", "TP"),
    ("BIOTECNOL", "TB"),
    ("ALIMENTOS", "TPIA"),
    ("ADMINISTRA", "TAE"),
    ("EMPRENDIM", "TAE"),
    ("AGROPECUAR", "TA"),
]

def _limpia_carrera(val) -> str:
    """Mayúsculas, sin acentos ni caracteres corruptos (mojibake del sistema escolar)."""
    s = unicodedata.normalize("NFKD", str(val))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Z]", "", s.upper())

def sigla_carrera(val):
    k = _limpia_carrera(val)
    for frag, sig in SIGLAS_PATRON:
        if frag in k:
            return sig
    return None  # p. ej. 'Componente Básico Propedéutico' (1er semestre)

def etiqueta_grupo(row):
    """'5A TP' (el nombre del grupo ya incluye el grado). Primer semestre
    (Componente Básico Propedéutico): sin especialidad."""
    base = str(row["grupo"]).strip()
    try:
        sem = int(row["semestre"])
    except (TypeError, ValueError):
        sem = None
    if sem == 1 or pd.isna(row.get("carrera")):
        return base
    sig = sigla_carrera(row["carrera"])
    return f"{base} {sig}" if sig else base

# ---------------------------------------------------------------- Base y modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL)

def query(sql, params=()):
    con = sqlite3.connect(DB)
    df = pd.read_sql(sql, con, params=params)
    con.close()
    return df

def ejecutar(sql, params=()):
    con = sqlite3.connect(DB)
    con.execute(sql, params)
    con.commit()
    con.close()

# Asegurar que existan tablas creadas en caliente por la app (si la base
# viene del ETL no existen hasta que se usan por primera vez).
if os.path.exists(DB):
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS intervenciones(
                   id INTEGER PRIMARY KEY AUTOINCREMENT, no_control TEXT, tipo TEXT,
                   comentario TEXT, usuario TEXT, fecha TEXT)""")
    try:
        # Marca de ciclo en calificacion: 'anterior' (ETL) vs 'actual' (parciales subidos)
        con.execute("ALTER TABLE calificacion ADD COLUMN ciclo TEXT")
        con.commit()
    except sqlite3.OperationalError:
        pass  # la columna ya existe
    con.close()

# ---------------------------------------------------------------- Autenticación
def login(usuario, password):
    h = hashlib.sha256(password.encode()).hexdigest()
    u = query("SELECT * FROM usuarios WHERE usuario=? AND pass_hash=?", (usuario, h))
    if len(u) == 1:
        return u.iloc[0].to_dict()
    return None

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.title("SARTIA")
        st.caption("Sistema de Alerta Temprana de Riesgo Académico · CBTA 197")
        st.info("Acceso solo para personal autorizado. Datos protegidos conforme a la LGPDPPSO.")
        with st.form("login"):
            usuario = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Entrar", use_container_width=True):
                u = login(usuario.strip(), password)
                if u:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
    st.stop()

user = st.session_state.user
ES_TUTOR = user["rol"] == "tutor"

# ---------------------------------------------------------------- Menú lateral
st.sidebar.title("SARTIA")
st.sidebar.caption(f"{user['usuario']} · {'Tutor' if ES_TUTOR else 'Coordinación'}")
if ES_TUTOR:
    opciones = ["Semáforo de riesgo", "Ficha de alumno", "Exportar reporte"]
else:
    opciones = ["Panel general", "Semáforo de riesgo", "Ficha de alumno",
                "Cargar parcial", "Modelos y métricas", "Exportar reporte"]
menu = st.sidebar.radio("Menú", opciones)
st.sidebar.markdown("---")
if st.sidebar.button("Cerrar sesión"):
    st.session_state.user = None
    st.rerun()

# ---------------------------------------------------------------- Datos
riesgo = query("""SELECT r.*, a.nombre, a.grupo, a.generacion, a.semestre, a.carrera,
                  p.paraescolar_total, s.horas_acumuladas
                  FROM riesgo r
                  JOIN alumno a ON a.no_control = r.no_control
                  LEFT JOIN paraescolares p ON p.no_control = r.no_control
                  LEFT JOIN servicio_social s ON s.no_control = r.no_control""")
# Solo alumnos del ciclo ACTUAL (los que aparecen en los parciales subidos).
# Si aún no se ha cargado nada en el ciclo, se muestra el baseline completo.
_activos = set(query("SELECT DISTINCT no_control FROM calificacion WHERE ciclo='actual'")["no_control"])
if _activos:
    riesgo = riesgo[riesgo["no_control"].isin(_activos)]
riesgo["etiqueta"] = riesgo.apply(etiqueta_grupo, axis=1)
if ES_TUTOR:
    riesgo = riesgo[riesgo["etiqueta"] == user["grupo_asignado"]]

def nivel_color(n):
    return {"Rojo": "🔴", "Amarillo": "🟡", "Verde": "🟢"}.get(n, "⚪")

# ================================================================ P02 PANEL GENERAL
if menu == "Panel general" and not ES_TUTOR:
    st.title("Panel general")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alumnos en seguimiento", len(riesgo))
    c2.metric("🔴 Riesgo alto", int((riesgo["nivel"] == "Rojo").sum()))
    c3.metric("🟡 Riesgo medio", int((riesgo["nivel"] == "Amarillo").sum()))
    c4.metric("🟢 Sin riesgo", int((riesgo["nivel"] == "Verde").sum()))
    g = riesgo.copy()
    g["etiqueta"] = g.apply(etiqueta_grupo, axis=1)
    gg = g.groupby(["etiqueta", "nivel"]).size().unstack(fill_value=0)
    for col in ["Rojo", "Amarillo", "Verde"]:
        if col not in gg.columns:
            gg[col] = 0
    st.subheader("Riesgo acumulado por grupo")
    st.bar_chart(gg[["Rojo", "Amarillo", "Verde"]])

    # --- Indicadores del parcial en curso (solo lo cargado en ESTE ciclo) ---
    cal = query("SELECT no_control, materia, p1, p2, p3, a1, a2, a3, max1 "
                "FROM calificacion WHERE ciclo='actual'")
    partes = []
    for k in (1, 2, 3):
        pk, ak = f"p{k}", f"a{k}"
        sub = cal.loc[cal[pk].notna(), ["no_control", "materia", pk, ak, "max1"]].copy()
        if sub.empty:
            continue
        sub.columns = ["no_control", "materia", "calif", "asis", "max1"]
        sub["parcial"] = k
        partes.append(sub)
    if partes:
        L = pd.concat(partes, ignore_index=True)
        L["reprobada"] = L["calif"] < 7
        L["asist_pct"] = (100 * L["asis"] / L["max1"]).clip(upper=100)
        indice = 100 * L["reprobada"].mean()
        parciales_txt = ", ".join(str(p) for p in sorted(L["parcial"].unique()))
        st.subheader(f"Indicadores del parcial en curso (parcial {parciales_txt})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registros evaluados", f"{len(L):,}")
        m2.metric("Índice de reprobación", f"{indice:.1f} %")
        m3.metric("Asistencia promedio", f"{L['asist_pct'].mean():.0f} %")
        m4.metric("Alumnos con ≥1 reprobada", int(L.loc[L["reprobada"], "no_control"].nunique()))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Índice de reprobación por parcial")
            idx_p = (100 * L.groupby("parcial")["reprobada"].mean()).rename("índice %")
            st.bar_chart(idx_p)
        with c2:
            st.subheader("Top materias reprobadas (parcial en curso)")
            top = (L[L["reprobada"]].groupby("materia").size()
                   .sort_values(ascending=False).head(10).rename("reprobadas"))
            st.bar_chart(top)
    else:
        st.info("Aún no se ha cargado ningún parcial de este ciclo. Usa el menú "
                "'Cargar parcial' para alimentar los indicadores.")
    st.subheader("Evidencia: señales tempranas (parcial 1)")
    st.table(pd.DataFrame({
        "Señales tempranas": ["0 señales", "1 señal", "2 señales", "3 señales"],
        "Tasa de reprobación observada": ["0.2 %", "4.6 %", "32.8 %", "65.2 %"],
    }).set_index("Señales tempranas"))
    st.caption("Fuente: análisis exploratorio del ciclo anterior (calificaciones, asistencia normalizada y paraescolares).")

# ================================================================ P03 SEMAFORO
elif menu == "Semáforo de riesgo":
    titulo_semaforo = "Semáforo de riesgo"
    if ES_TUTOR:
        titulo_semaforo += " — Grupo " + user["grupo_asignado"]
    st.title(titulo_semaforo)
    f1, f2, f3 = st.columns(3)
    df = riesgo.copy()
    etiquetas = ["Todos"] + sorted(df["etiqueta"].unique())
    fg = f1.selectbox("Grupo", etiquetas if not ES_TUTOR else ["Todos"])
    fn = f2.selectbox("Nivel", ["Todos", "Rojo", "Amarillo", "Verde"])
    texto = f3.text_input("Buscar nombre o número de control")
    if fg != "Todos":
        df = df[df["etiqueta"] == fg]
    if fn != "Todos":
        df = df[df["nivel"] == fn]
    if texto:
        df = df[df["nombre"].str.contains(texto, case=False) | df["no_control"].str.contains(texto)]
    if df.empty:
        st.warning("No se encontraron alumnos con esos filtros.")
    else:
        df = df.sort_values("prob_riesgo", ascending=False)
        viz = df[["no_control", "nombre", "etiqueta", "nivel", "prob_riesgo",
                  "paraescolar_total", "horas_acumuladas"]].copy()
        viz[" "] = viz["nivel"].map(nivel_color)
        viz["prob_riesgo"] = (viz["prob_riesgo"] * 100).round(0).astype(int).astype(str) + " %"
        viz = viz.rename(columns={"no_control": "No. control", "nombre": "Nombre", "etiqueta": "Grupo",
                                  "nivel": "Nivel", "prob_riesgo": "Probabilidad",
                                  "paraescolar_total": "Puntos paraescolares",
                                  "horas_acumuladas": "Horas de SS"})
        st.dataframe(viz, use_container_width=True, hide_index=True)
        ultima = riesgo["fecha"].max() if "fecha" in riesgo.columns else ""
        st.caption(f"{len(df)} alumnos · modelo: Gradient Boosting + SMOTE · "
                   f"última actualización: {ultima}")

# ================================================================ P04 FICHA
elif menu == "Ficha de alumno":
    st.title("Ficha de alumno")
    sel = st.selectbox("Alumno", riesgo.sort_values("prob_riesgo", ascending=False)["no_control"] + " — " + riesgo.sort_values("prob_riesgo", ascending=False)["nombre"])
    nc = sel.split(" — ")[0]
    # Verificación de permiso en capa de datos
    alumno = riesgo[riesgo["no_control"] == nc]
    if len(alumno) == 0:
        st.error("No autorizado: el alumno no pertenece a tu grupo asignado.")
        st.stop()
    a = alumno.iloc[0]
    cal = query("SELECT materia, p1, p2, p3, a1, a2, a3, max1 FROM calificacion "
                "WHERE no_control=? AND ciclo='actual'", (nc,))
    hist = query("SELECT p1, p2, p3, a1, a2, a3, max1 FROM calificacion "
                 "WHERE no_control=? AND (ciclo IS NULL OR ciclo <> 'actual')", (nc,))
    antecedentes = []
    if len(hist):
        n_rep = int((hist[["p1", "p2", "p3"]] < 7).sum().sum())
        asist_h = pd.concat([100 * hist[f"a{k}"] / hist["max1"] for k in (1, 2, 3)], axis=1)
        n_asist = int((asist_h < 80).sum().sum())
        if n_rep >= 2:
            antecedentes.append(
                f"antecedentes de calificaciones bajas en ciclos anteriores ({n_rep} parciales < 7)")
        if n_asist >= 3:
            antecedentes.append(
                f"antecedentes de asistencia baja en ciclos anteriores ({n_asist} registros < 80 %)")
    # Promedio y asistencia del parcial más reciente con datos (3 -> 1)
    prom_p1, asist_min = np.nan, np.nan
    for k in (3, 2, 1):
        pk, ak = f"p{k}", f"a{k}"
        if pk in cal.columns and cal[pk].notna().any():
            sub = cal[cal[pk].notna()]
            prom_p1 = sub[pk].mean()
            if ak in sub.columns and sub[ak].notna().any():
                asist_min = (100 * sub.loc[sub[ak].notna(), ak] /
                             sub.loc[sub[ak].notna(), "max1"]).min()
            break
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Probabilidad de riesgo", f"{a['prob_riesgo']*100:.0f} %")
        st.markdown(f"### {nivel_color(a['nivel'])} Nivel: **{a['nivel']}**")
        st.caption(f"Umbral: rojo ≥ 40 % · amarillo ≥ 8 %")
    with c2:
        st.subheader("¿Por qué está en riesgo?")
        causas = list(antecedentes)
        if pd.notna(a["paraescolar_total"]) and a["paraescolar_total"] == 0:
            causas.append("no acumula puntos de actividades paraescolares (0 pts)")
        if pd.notna(prom_p1) and prom_p1 < 7:
            causas.append(f"promedio de {prom_p1:.1f} en el último parcial con datos")
        if pd.notna(asist_min) and asist_min < 80:
            causas.append(f"asistencia mínima de {asist_min:.0f}% (umbral institucional 80%)")
        if pd.notna(a["horas_acumuladas"]) and a["horas_acumuladas"] < 480 and a["semestre"] == 6:
            causas.append(f"servicio social retrasado ({a['horas_acumuladas']:.0f} de 480 h)")
        if causas:
            st.info("El alumno presenta **" + a["nivel"].upper() + "** porque: " + ", ".join(causas) + ".")
        else:
            st.success("Sin factores de riesgo destacados en este corte.")
    st.subheader("Materias en riesgo")
    mostrar_todas = st.checkbox("Mostrar todas las materias", value=False,
                                help="Por defecto solo se listan las materias con "
                                     "calificación < 7 o asistencia < 80 %")
    if len(cal):
        filas = []
        for _, r in cal.iterrows():
            for k in (1, 2, 3):
                if pd.notna(r.get(f"p{k}")):
                    pct = (100 * r[f"a{k}"] / r["max1"]) if pd.notna(r.get(f"a{k}")) and r["max1"] else np.nan
                    filas.append({"Materia": r["materia"], "Parcial": k,
                                  "Calificación": r[f"p{k}"],
                                  "Asistencia %": min(round(pct), 100) if pd.notna(pct) else None})
        det = pd.DataFrame(filas)
        if len(det):
            en_riesgo = det[(det["Calificación"] < 7) |
                            (det["Asistencia %"].notna() & (det["Asistencia %"] < 80))]
            if mostrar_todas:
                st.dataframe(det, use_container_width=True, hide_index=True)
            elif len(en_riesgo):
                st.dataframe(en_riesgo, use_container_width=True, hide_index=True)
            else:
                st.success("Ninguna materia en riesgo: todas con calificación ≥ 7 "
                           "y asistencia ≥ 80 %.")
    st.subheader("Registro de intervención")
    with st.form("intervencion"):
        c1, c2 = st.columns([1, 2])
        tipo = c1.selectbox("Tipo *", ["", "Tutoría individual", "Orientación educativa",
                                       "Citación a padres de familia", "Canalización a psicología"])
        coment = c2.text_input("Comentario (opcional)")
        if st.form_submit_button("Guardar"):
            if not tipo:
                st.error("Selecciona el tipo de intervención antes de guardar.")
            else:
                ejecutar("INSERT INTO intervenciones(no_control,tipo,comentario,usuario,fecha) VALUES(?,?,?,?,?)",
                         (nc, tipo, coment, user["usuario"], dt.datetime.now().isoformat(timespec="seconds")))
                st.success("Intervención registrada.")
    hist = query("SELECT fecha, usuario, tipo, comentario FROM intervenciones WHERE no_control=? ORDER BY fecha DESC", (nc,))
    if len(hist):
        st.dataframe(hist, use_container_width=True, hide_index=True)

# ================================================================ P05 CARGAR PARCIAL
elif menu == "Cargar parcial":
    st.title("Cargar calificaciones del parcial")
    st.caption("Única entrada manual del sistema. Paraescolares y servicio social se "
               "sincronizan desde Google Sheets. Las features del modelo son acumulativas "
               "hasta el parcial cargado.")
    parcial = st.selectbox("Parcial a cargar", [1, 2, 3],
                           help="Parcial 1 ≈ semana 6 · Parcial 2 ≈ semana 11 · Parcial 3 ≈ semana 16")
    arch = st.file_uploader("Archivo exportado del sistema escolar (.xls / .xlsx)", type=["xls", "xlsx"])
    if arch:
        try:
            raw = arch.read()
            try:
                df = pd.read_excel(__import__("io").BytesIO(raw))
            except Exception:
                df = pd.read_html(__import__("io").BytesIO(raw), encoding="iso-8859-1")[0]
        except Exception:
            st.error("No se pudo leer el archivo. Exporta de nuevo desde el sistema escolar.")
            st.stop()
        col_nc, col_mat = None, None
        cols_cal, cols_as = {}, {}   # {periodo: columna} para periodos 1..parcial
        for c in df.columns:
            cn = norm(c)
            if "CONTROL" in cn:
                col_nc = c
            elif "ASIGNATURA" in cn:
                col_mat = c
            else:
                for i in range(1, parcial + 1):
                    if cn == f"PERIODO{i}" or (i == 1 and cn == "P1"):
                        cols_cal[i] = c
                    elif cn == f"ASISTENCIAPERIODO{i}" or (i == 1 and cn == "A1"):
                        cols_as[i] = c
        if not col_nc or not col_mat or parcial not in cols_cal or parcial not in cols_as:
            st.error(f"Formato inválido: no se encontraron las columnas del parcial {parcial} "
                     f"(control, materia, periodo {parcial}, asistencia {parcial}). "
                     f"Columnas recibidas: {list(df.columns)[:8]}...")
            st.stop()
        df = df.rename(columns={col_nc: "no_control", col_mat: "materia"})
        df["no_control"] = df["no_control"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        # Calificaciones acumuladas 1..parcial ("AC" -> NaN, "NAC" -> 5.0, coherente con el ETL)
        P = pd.DataFrame({i: pd.to_numeric(df[c].replace({"AC": np.nan, "NAC": 5.0}), errors="coerce")
                          for i, c in cols_cal.items()})
        # Asistencia acumulada: suma de asistencias 1..parcial sobre parcial*max1
        A = pd.DataFrame({i: pd.to_numeric(df[c], errors="coerce")
                          for i, c in cols_as.items()})
        df["prom"] = P.mean(axis=1)
        df["minp"] = P.min(axis=1)
        df["max1"] = df["materia"].map(max_p1)
        n_per = len(cols_as)
        df["pct"] = (100 * A.sum(axis=1) / (n_per * df["max1"])).clip(upper=100)
        df = df.dropna(subset=["prom", "pct"], how="all")
        padron = set(query("SELECT no_control FROM alumno")["no_control"])
        reconocidos = df["no_control"].isin(padron).sum()
        st.subheader("Vista previa")
        st.write(f"**{len(df):,}** registros leídos · **{reconocidos:,}** alumnos reconocidos en el padrón · "
                 f"**{len(df) - reconocidos:,}** números de control no encontrados")
        st.dataframe(df[["no_control", "materia", "prom", "pct"]].head(8),
                     use_container_width=True, hide_index=True)
        if st.button("Procesar y recalcular predicciones", type="primary"):
            with st.spinner("Normalizando asistencia y calculando riesgos..."):
                feats = df.groupby("no_control").agg(
                    prom_p1=("prom", "mean"), min_p1=("minp", "min"),
                    asist_min_p1=("pct", "min"), asist_prom_p1=("pct", "mean"))
                feats = feats.reset_index().merge(
                    query("SELECT no_control, paraescolar_total FROM paraescolares"), on="no_control", how="left")
                feats = feats.merge(
                    query("SELECT no_control, horas_acumuladas FROM servicio_social"), on="no_control", how="left")
                para = query("SELECT paraescolar_total FROM paraescolares")["paraescolar_total"].mean()
                hrs = query("SELECT horas_acumuladas FROM servicio_social")["horas_acumuladas"].mean()
                feats["paraescolar_total"] = feats["paraescolar_total"].fillna(para if pd.notna(para) else 15)
                feats["horas_acumuladas"] = feats["horas_acumuladas"].fillna(hrs if pd.notna(hrs) else 240)
                # El modelo espera los nombres con que fue entrenado (entrenar_modelo.py)
                feats = feats.rename(columns={"paraescolar_total": "para_pts",
                                              "horas_acumuladas": "horas_ss"})
                pack = cargar_modelo()
                faltan = [c for c in pack["cols"] if c not in feats.columns]
                if faltan:
                    st.error(f"El modelo espera columnas que no se pudieron construir: {faltan}")
                    st.stop()
                Xn = feats[pack["cols"]].values.astype(float)
                proba = pack["modelo"].predict_proba(Xn)[:, 1]
                hoy = dt.date.today().isoformat()
                pk, ak = f"p{parcial}", f"a{parcial}"
                con = sqlite3.connect(DB)
                # 1) Guardar el parcial en la tabla calificacion (historial del ciclo).
                #    Al subir el parcial k se BORRAN los parciales k+1..3: en el
                #    ciclo en curso aún no existen y el upsert heredaba los del
                #    ciclo anterior (misma llave alumno-materia).
                cols_null = []
                for kk in range(parcial + 1, 4):
                    cols_null += [f"p{kk}=NULL", f"a{kk}=NULL"]
                limpia = (", " + ", ".join(cols_null)) if cols_null else ""
                for _, r in df.iterrows():
                    con.execute(f"""INSERT INTO calificacion(no_control, materia, {pk}, {ak}, max1, ciclo)
                                    VALUES(?,?,?,?,?,'actual')
                                    ON CONFLICT(no_control, materia) DO UPDATE SET
                                    {pk}=excluded.{pk}, {ak}=excluded.{ak}, ciclo=excluded.ciclo{limpia}""",
                                (r["no_control"], r["materia"],
                                 None if pd.isna(P.loc[r.name, parcial]) else float(P.loc[r.name, parcial]),
                                 None if pd.isna(A.loc[r.name, parcial]) else float(A.loc[r.name, parcial]),
                                 float(r["max1"])))
                # 2) Actualizar el padrón de alumnos (grado, grupo, carrera,
                #    generación y nombre) con los datos del archivo subido
                info = {norm(c): c for c in df.columns}
                def _col(*names):
                    return next((info[n] for n in names if n in info), None)
                c_grp = _col("GRUPO")
                c_sem = _col("SEMESTRE")
                c_gen = _col("GENERACION")
                c_car = _col("CARRERA")
                c_nom = [_col("NOMBRE"), _col("PATERNO"), _col("MATERNO")]
                for _, r in df.drop_duplicates("no_control").iterrows():
                    nombre = " ".join(str(r[c]).strip() for c in c_nom
                                      if c and pd.notna(r[c]) and str(r[c]).strip() not in ("", "nan"))
                    generacion = None
                    if c_gen and pd.notna(r[c_gen]):
                        generacion = re.sub(r"(\d{4})\s*-\s*(\d{4})", r"\1-\2",
                                            str(r[c_gen]).strip())
                    semestre = pd.to_numeric(r[c_sem], errors="coerce") if c_sem else np.nan
                    con.execute("""INSERT INTO alumno(no_control, nombre, grupo, generacion, semestre, carrera)
                                   VALUES(?,?,?,?,?,?)
                                   ON CONFLICT(no_control) DO UPDATE SET
                                   nombre=COALESCE(NULLIF(excluded.nombre,''), alumno.nombre),
                                   grupo=COALESCE(NULLIF(excluded.grupo,''), alumno.grupo),
                                   generacion=COALESCE(NULLIF(excluded.generacion,''), alumno.generacion),
                                   semestre=COALESCE(excluded.semestre, alumno.semestre),
                                   carrera=COALESCE(NULLIF(excluded.carrera,''), alumno.carrera)""",
                                (r["no_control"], nombre,
                                 str(r[c_grp]).strip() if c_grp and pd.notna(r[c_grp]) else None,
                                 generacion,
                                 int(semestre) if pd.notna(semestre) else None,
                                 str(r[c_car]).strip() if c_car and pd.notna(r[c_car]) else None))
                # 2b) Sincronizar cuentas de tutor con los grupos del CICLO
                #     ACTUAL (el archivo subido es la foto del plantel):
                #     crea las que faltan con contraseña única y elimina las
                #     de grupos que ya no existen en este ciclo.
                etq_src = pd.DataFrame({
                    "grupo": df[c_grp].astype(str).str.strip() if c_grp else "",
                    "semestre": pd.to_numeric(df[c_sem], errors="coerce") if c_sem else np.nan,
                    "carrera": df[c_car].astype(str).str.strip() if c_car else "",
                }).drop_duplicates()
                etqs = sorted({e for e in (etiqueta_grupo(r) for _, r in etq_src.iterrows()) if e})
                nuevas = []
                for etq in etqs:
                    usuario = "tutor." + etq.lower().replace(" ", "-")
                    if con.execute("SELECT 1 FROM usuarios WHERE usuario=?",
                                   (usuario,)).fetchone():
                        continue
                    pwd = _nueva_password()
                    con.execute("INSERT INTO usuarios(usuario, pass_hash, rol, grupo_asignado)"
                                " VALUES(?,?,?,?)",
                                (usuario, hashlib.sha256(pwd.encode()).hexdigest(),
                                 "tutor", etq))
                    nuevas.append((usuario, pwd, etq))
                if etqs:
                    ph = ",".join("?" * len(etqs))
                    con.execute(f"DELETE FROM usuarios WHERE rol='tutor' AND "
                                f"grupo_asignado NOT IN ({ph})", etqs)
                if nuevas:
                    with open("data/cuentas_tutores.csv", "a", encoding="utf-8") as f:
                        for u, p, e in nuevas:
                            f.write(f"{u},{p},{e}\n")
                # 3) Actualizar el semáforo de riesgo
                for i, row in feats.iterrows():
                    nivel = "Rojo" if proba[i] >= 0.40 else ("Amarillo" if proba[i] >= 0.08 else "Verde")
                    con.execute("""INSERT INTO riesgo(no_control,semana,prob_riesgo,nivel,modelo,fecha)
                                   VALUES(?,?,?,?,?,?)
                                   ON CONFLICT(no_control) DO UPDATE SET
                                   semana=excluded.semana, prob_riesgo=excluded.prob_riesgo,
                                   nivel=excluded.nivel, fecha=excluded.fecha""",
                                (row["no_control"], SEMANA_POR_PARCIAL[parcial], float(proba[i]), nivel,
                                 f"GradientBoosting+SMOTE (parcial {parcial})", hoy))
                con.commit()
                con.close()
            st.success(f"Parcial {parcial} cargado: {len(feats):,} alumnos procesados · "
                       f"{int((proba>=0.40).sum())} en rojo · {hoy}")
            if nuevas:
                st.warning(f"Se crearon {len(nuevas)} cuenta(s) de tutor nueva(s): "
                           + ", ".join(u for u, _, _ in nuevas)
                           + ". Contraseñas en data/cuentas_tutores.csv")
            st.balloons()
            st.info("El semáforo y el panel general ya muestran las predicciones actualizadas.")

# ================================================================ P06 MODELOS
elif menu == "Modelos y métricas":
    st.title("Modelos y métricas")
    st.caption("Validación cruzada estratificada de 10 pliegues · SMOTE solo en entrenamiento · "
               "entrenado con los datos reales del ciclo anterior.")
    met = query("SELECT * FROM metricas_modelos").set_index("modelo")
    st.dataframe(met, use_container_width=True)
    st.bar_chart(met[["recall", "precision", "f1"]])
    st.subheader("Lectura")
    st.success("El modelo de producción se entrena con las calificaciones finales del ciclo "
               "anterior (etiqueta: no acreditada) y se alimenta durante el ciclo actual con "
               "los parciales 1, 2 y 3 a medida que se cargan.")
    st.subheader("Importancia de variables (permutación, scoring=recall)")
    imp = pd.DataFrame({"variable": ["prom_p1", "min_p1", "para_pts", "asist_min_p1", "asist_prom_p1", "horas_ss"],
                        "importancia": [0.85, 0.79, 0.81, 0.52, 0.18, 0.30]}).set_index("variable")
    st.bar_chart(imp)
    st.caption("Pendiente en la versión final: valores SHAP locales por alumno y XGBoost en sustitución de Gradient Boosting.")

# ================================================================ P07 EXPORTAR
elif menu == "Exportar reporte":
    st.title("Exportar lista de alertados")
    minimo = st.selectbox("Nivel mínimo", ["Rojo", "Amarillo", "Verde"])
    orden = {"Rojo": 0, "Amarillo": 1, "Verde": 2}
    df = riesgo[riesgo["nivel"].map(orden) <= orden[minimo]].sort_values("prob_riesgo", ascending=False)
    if df.empty:
        st.warning("No hay alumnos en ese nivel; ajusta el filtro.")
    else:
        st.write(f"**{len(df)}** alumnos incluidos")
        st.dataframe(df[["no_control", "nombre", "grupo", "nivel", "prob_riesgo"]],
                     use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Descargar CSV", csv, f"sartia_alertados_{dt.date.today()}.csv", "text/csv")
        st.caption("El archivo puede imprimirse como evidencia de las citaciones a tutoría.")
