# -*- coding: utf-8 -*-
"""
ETL SARTIA — Integración de las 3 fuentes del CBTA 197 en SQLite.

Fuentes:
  1. detalle_calificaciones*.xls  -> realmente es HTML (exportado del sistema
     de control escolar). Se lee con pd.read_html. Es formato LARGO:
     una fila por alumno-por-asignatura con PERIODO 1/2/3.
  2. estatus servicio social feb-jul2026.xlsx, hoja 'captura'.
  3. creditos formacion academica feb-jul 2026.xlsx, hoja 'Creditos'.

Salida: data/sartia.db con tablas alumno, riesgo, calificacion, paraescolares,
servicio_social, usuarios, metricas_modelos e intervenciones, con no_control
(TEXT, 14 dígitos) como llave primaria, lista para el dashboard de app.py.

La tabla calificacion guarda los 3 parciales y la calificación final del
ciclo: sirve como HISTORIAL de referencia y como fuente de entrenamiento
del modelo (entrenar_modelo.py usa p1 + la final como etiqueta).

Uso:
    python etl_sartia.py --dir "BD" --out data/sartia.db
"""
import argparse
import datetime as dt
import hashlib
import os
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de rutas de entrada
# ---------------------------------------------------------------------------
# Se busca por PATRÓN, no por nombre exacto: cada exportación del sistema
# escolar genera un nombre distinto (p. ej. con fecha/hora), así que basta
# con que el archivo empiece igual y esté en la carpeta indicada con --dir.
PATRONES = {
    "calificaciones": ("detalle_calificaciones", (".xls", ".xlsx", ".html")),
    "servicio_social": ("estatus servicio social", (".xlsx", ".xls")),
    "creditos": ("creditos formacion academica", (".xlsx", ".xls")),
}


def resolver_rutas(carpeta):
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(
            f"No existe la carpeta de datos: {carpeta}\n"
            f"Directorio actual: {os.getcwd()}"
        )
    archivos = sorted(os.listdir(carpeta))
    rutas = {}
    for clave, (inicio, exts) in PATRONES.items():
        candidatos = [f for f in archivos
                      if f.lower().startswith(inicio) and f.lower().endswith(exts)]
        if not candidatos:
            raise FileNotFoundError(
                f"No se encontró el archivo de {clave!r} "
                f"(debe empezar con {inicio!r}) en: {os.path.abspath(carpeta)}")
        rutas[clave] = os.path.join(carpeta, candidatos[-1])  # el más reciente
    return rutas

CAL_PERIODO_COLS = ["PERIODO 1", "PERIODO 2", "PERIODO 3"]
CAL_ASIS_COLS = ["ASISTENCIA PERIODO 1", "ASISTENCIA PERIODO 2", "ASISTENCIA PERIODO 3"]
UMBRAL_REPROBATORIA = 6.0          # < 6 o NAC = no acreditada
HORAS_REQUERIDAS_SS = 480.0        # servicio social (según anteproyecto)

# Tasas de reprobación observadas por señales (tabla de evidencia del dashboard)
PROB_POR_REPROBADAS = {0: 0.002, 1: 0.046, 2: 0.328}
PROB_3_O_MAS = 0.652
UMBRAL_ROJO, UMBRAL_AMARILLO = 0.40, 0.08

# Catálogo de sesiones máximas por materia (parcial 1) — transcrito del PDF del
# plantel. Se usa para normalizar la asistencia (a1/max1) en la tabla calificacion.
def _norm_txt(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", s.upper())

CATALOGO_P1 = {
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
    k = _norm_txt(materia)
    for clave, mx in CATALOGO_P1.items():
        if clave in k or k in clave:
            return mx
    return 24  # valor típico si la materia no está en el catálogo

# Limpieza de nombres de carrera con codificación rota en el sistema escolar
MAPA_CARRERAS = {
    "TÉCNICO AGROPECUARIO": "TÉCNICO AGROPECUARIO",
    "TÉCNICO EN PROGRAMACIÓN": "TÉCNICO EN PROGRAMACIÓN",
    "TÉCNICO EN PRODUCCION INDUSTRIAL DE ALIMENTOS":
        "TÉCNICO EN PRODUCCIÓN INDUSTRIAL DE ALIMENTOS",
    "TÉCNICO EN BIOTECNOLOGÍA": "TÉCNICO EN BIOTECNOLOGÍA",
    "TÉCNICO EN ADMINISTRACIÓN PARA EL EMPRENDIMIENTO AGROPECUARIO":
        "TÉCNICO EN ADMINISTRACIÓN PARA EL EMPRENDIMIENTO AGROPECUARIO",
    "TÉCNICO EN ADMINISTRACIÓN DE EMPRENDIMIENTOS":
        "TÉCNICO EN ADMINISTRACIÓN PARA EL EMPRENDIMIENTO AGROPECUARIO",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def norm_no_control(series: pd.Series) -> pd.Series:
    """no_control normalizado: texto, sin espacios, 14 dígitos (rellena con 0)."""
    s = (series.astype(str)
           .str.strip()
           .str.replace(r"\.0$", "", regex=True)
           .str.replace(r"\s+", "", regex=True))
    s = s.where(s.str.fullmatch(r"\d+"), other=np.nan)
    return s.str.zfill(14)


def norm_generacion(val: str) -> str:
    if pd.isna(val):
        return None
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", str(val))
    return f"{m.group(1)}-{m.group(2)}" if m else str(val).strip()


def norm_carrera(val: str) -> str:
    if pd.isna(val):
        return None
    clave = str(val).strip().upper()
    clave = clave.replace("ó", "O").replace("Ó", "O").replace("í", "I").replace("Í", "I")
    for k, v in MAPA_CARRERAS.items():
        if k.replace("Ó", "O").replace("Í", "I") == clave:
            return v
    return str(val).strip().upper()


def reprobada(serie: pd.Series) -> pd.Series:
    """True si la asignatura NO está acreditada en ese parcial."""
    num = pd.to_numeric(serie, errors="coerce")
    es_nac = serie.astype(str).str.strip().str.upper().eq("NAC")
    return (num < UMBRAL_REPROBATORIA).fillna(False) | es_nac


def acreditado_a_bool(val) -> float:
    """Convierte 'Acreditado'/'True'/True/5 a 1, 'NO'/NaN a 0 (NaN si no aplica)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return 1.0 if val > 0 else 0.0
    t = str(val).strip().upper()
    if t in ("ACREDITADO", "TRUE", "SÍ", "SI"):
        return 1.0
    return 0.0


def p_numerico(serie: pd.Series) -> pd.Series:
    """Calificación de parcial numérica: 'AC' (acreditada sin calif.) -> NaN,
    'NAC' (no acreditada) -> 5.0, para que los promedios sean coherentes."""
    return pd.to_numeric(serie.replace({"AC": np.nan, "NAC": 5.0}), errors="coerce")


# ---------------------------------------------------------------------------
# 1. Calificaciones  (HTML largo: una fila por alumno-asignatura)
# ---------------------------------------------------------------------------
def cargar_calificaciones(path: str):
    df = pd.read_html(path)[0]
    df["no_control"] = norm_no_control(df["NO CONTROL"])
    df = df.dropna(subset=["no_control"])

    # Nombre completo
    df["nombre"] = (df["NOMBRE"].fillna("") + " " +
                    df["PATERNO"].fillna("") + " " +
                    df["MATERNO"].fillna("")).str.replace(r"\s+", " ", regex=True).str.strip()

    # Numéricos de parciales y asistencia
    for c in CAL_PERIODO_COLS + CAL_ASIS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Tabla alumno (una fila por alumno) ----
    g = df.groupby("no_control")
    alumno = pd.DataFrame({
        "nombre": g["nombre"].first(),
        "grupo": g["GRUPO"].first(),
        "generacion": g["GENERACION"].first().map(norm_generacion),
        "semestre": g["SEMESTRE"].agg(lambda s: s.mode().iloc[0]).astype(int),
        "carrera": g["CARRERA"].first().map(norm_carrera),
    }).reset_index()

    # ---- Tabla calificacion (largo: una fila por alumno-materia, 3 parciales) ----
    # Guarda el historial completo del ciclo: p1..p3, a1..a3, final y tipo de
    # acreditación. Es la fuente de entrenamiento del modelo (ver entrenar_modelo.py).
    calificacion = pd.DataFrame({
        "no_control": df["no_control"],
        "materia": df["NOMBRE ASIGNATURA"].astype(str).str.strip(),
        "p1": p_numerico(df["PERIODO 1"]),
        "a1": df["ASISTENCIA PERIODO 1"],
        "p2": p_numerico(df["PERIODO 2"]),
        "a2": df["ASISTENCIA PERIODO 2"],
        "p3": p_numerico(df["PERIODO 3"]),
        "a3": df["ASISTENCIA PERIODO 3"],
        # El catálogo solo cubre sesiones del parcial 1; se usa como aproximación
        # para normalizar los parciales 2 y 3.
        "max1": df["NOMBRE ASIGNATURA"].map(max_p1),
        "max2": df["NOMBRE ASIGNATURA"].map(max_p1),
        "max3": df["NOMBRE ASIGNATURA"].map(max_p1),
        "final": pd.to_numeric(df.get("CALIFICACION"), errors="coerce"),
        "tipo_acred": df.get("TIPO ACRED.").astype(str).str.strip()
                      if "TIPO ACRED." in df.columns else None,
        # Marca de ciclo: lo que carga el ETL es historial de ciclos anteriores.
        # Los parciales subidos desde la app se marcan 'actual' (ver app.py).
        "ciclo": "anterior",
    }).drop_duplicates(subset=["no_control", "materia"])

    # ---- Tabla riesgo (features por parcial + baseline de reglas) ----
    # prob_riesgo usa las tasas de reprobación observadas por # de materias
    # reprobadas en P1 (tabla de evidencia del dashboard); umbral rojo >= 40 %.
    filas = []
    for nc, sub in g:
        row = {"no_control": nc}
        for i, (pc, ac) in enumerate(zip(CAL_PERIODO_COLS, CAL_ASIS_COLS), start=1):
            materias = sub[pc].notna() | sub[pc].astype(str).str.upper().eq("NAC")
            n_mat = int(materias.sum())
            rep = int((reprobada(sub[pc]) & materias).sum())
            row[f"materias_p{i}"] = n_mat
            row[f"reprobadas_p{i}"] = rep
            row[f"promedio_p{i}"] = round(float(sub.loc[materias, pc].mean()), 2) if n_mat else np.nan
            row[f"asistencia_media_p{i}"] = round(float(sub[ac].mean()), 2)
        rep1 = row["reprobadas_p1"]
        row["prob_riesgo"] = PROB_POR_REPROBADAS.get(rep1, PROB_3_O_MAS) if rep1 < 3 else PROB_3_O_MAS
        row["nivel"] = ("Rojo" if row["prob_riesgo"] >= UMBRAL_ROJO else
                        "Amarillo" if row["prob_riesgo"] >= UMBRAL_AMARILLO else "Verde")
        # Metadatos del corte (la app hace upsert sobre estas columnas)
        row["semana"] = 6
        row["modelo"] = "Baseline reglas (ETL)"
        row["fecha"] = dt.date.today().isoformat()
        filas.append(row)
    riesgo = pd.DataFrame(filas)
    return alumno, riesgo, calificacion


# ---------------------------------------------------------------------------
# 2. Servicio social  (hoja 'captura')
# ---------------------------------------------------------------------------
def cargar_servicio_social(path: str):
    df = pd.read_excel(path, sheet_name="captura")
    df["no_control"] = norm_no_control(df["control"])
    df = df.dropna(subset=["no_control"])

    # Filas fantasma: registros con "NO" en campos clave (alumnos dados de baja)
    mask_fantasma = (
        df["control"].astype(str).str.strip().str.upper().eq("NO") |
        df["NOMBRE"].astype(str).str.strip().str.upper().eq("NO"))
    df = df[~mask_fantasma].copy()

    entregas = ["ENTREGA 1", "ENTREGA 2", "ENTREGA 3", "EDNTREGA 4", "ENTREGA 5", "ENTREGA 6"]
    hrs_cols = ["HRS", "HRS.1", "HRS.2", "HRS.3", "HRS.4", "HRS.5"]

    for c in entregas + hrs_cols + ["TOTAL HRS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "no_control": df["no_control"],
        "horas_acumuladas": df["TOTAL HRS"].clip(lower=0),
        "entregas_realizadas": df[entregas].notna().sum(axis=1).astype(int),
        "horas_por_entrega_media": df[hrs_cols].mean(axis=1).round(1),
        "avance_ss": (df["TOTAL HRS"] / HORAS_REQUERIDAS_SS).clip(0, 1.5).round(3),
        "portafolio": df["PORTAFOLIO"].astype(str).str.strip().replace({"nan": None}),
        "carta": df["CARTA"].astype(str).str.strip().replace({"nan": None}),
        "constancia": df["CONSTANCIA"].astype(str).str.strip().replace({"nan": None}),
    })
    # Algunos no_control vienen duplicados en la captura: nos quedamos con el
    # registro con más horas (el más completo).
    out = out.sort_values("horas_acumuladas").drop_duplicates("no_control", keep="last")
    return out.sort_values("no_control").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Paraescolares / créditos de formación académica
# ---------------------------------------------------------------------------
def cargar_paraescolares(path: str):
    df = pd.read_excel(path, sheet_name="Creditos")
    df["no_control"] = norm_no_control(df["No. Control"])
    df = df.dropna(subset=["no_control"]).drop_duplicates("no_control", keep="first")

    col_actual = "PARA. FEB-JUL 2026"     # periodo en curso
    cols_hist = ["PARA. FEB-JUL 2025", "PARA. SEP DIC 2025"]

    out = pd.DataFrame({
        "no_control": df["no_control"],
        "tipo_paraescolar": df["PARAESCOLAR"].astype(str).str.strip().replace({"nan": None}),
        "actividad": df["ACTIVIDAD"].astype(str).str.strip().replace({"nan": None}),
        "paraescolar_acreditado_actual": df[col_actual].map(acreditado_a_bool)
            if col_actual in df.columns else np.nan,
        "paraescolar_total": pd.to_numeric(df.get("PARAESCOLAR TOTAL"),
                                           errors="coerce"),
    })
    for c in cols_hist:
        if c in df.columns:
            out[c.lower().replace(". ", "_").replace(" ", "_")] = df[c].map(acreditado_a_bool)
    return out.sort_values("no_control").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tablas de soporte del dashboard (login + métricas)
# ---------------------------------------------------------------------------
def _h(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Siglas de especialidad (mismo criterio que app.py para la etiqueta de grupo)
SIGLAS_CARRERA = {
    "TÉCNICO EN PROGRAMACIÓN": "TP",
    "TÉCNICO AGROPECUARIO": "TA",
    "TÉCNICO EN PRODUCCIÓN INDUSTRIAL DE ALIMENTOS": "TPIA",
    "TÉCNICO EN BIOTECNOLOGÍA": "TB",
    "TÉCNICO EN ADMINISTRACIÓN PARA EL EMPRENDIMIENTO AGROPECUARIO": "TAE",
}

def etiqueta_grupo(row) -> str:
    """'5A TP' (el nombre del grupo ya incluye el grado). 1er semestre: sin sigla."""
    base = str(row["grupo"]).strip()
    try:
        sem = int(row["semestre"])
    except (TypeError, ValueError):
        sem = None
    if sem == 1 or pd.isna(row.get("carrera")):
        return base
    sig = SIGLAS_CARRERA.get(str(row["carrera"]).strip().upper())
    return f"{base} {sig}" if sig else base

def _nueva_password(n: int = 10) -> str:
    """Contraseña aleatoria única (letras + dígitos)."""
    import secrets
    import string
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))

def cargar_usuarios(alumno: pd.DataFrame) -> pd.DataFrame:
    """Solo la cuenta del directivo. Las cuentas de tutor NO se crean aquí:
    se generan en la app al cargar el primer parcial del ciclo EN CURSO,
    una por cada grupo del archivo (grado+grupo+especialidad), con contraseña
    única aleatoria (ver sección 'Cargar parcial' de app.py). Así nunca hay
    cuentas de grupos de ciclos anteriores."""
    os.makedirs("data", exist_ok=True)
    # Reiniciar el CSV de credenciales (ciclo nuevo = credenciales nuevas)
    pd.DataFrame(columns=["usuario", "password", "grupo"]).to_csv(
        "data/cuentas_tutores.csv", index=False)
    return pd.DataFrame([
        {"usuario": "directivo", "pass_hash": _h("sartia2026"),
         "rol": "directivo", "grupo_asignado": None},
    ])

def cargar_metricas() -> pd.DataFrame:
    """Semilla de la tabla de métricas (se sobrescribe al entrenar el modelo)."""
    return pd.DataFrame([
        {"modelo": "Regla institucional (umbral 80%)",
         "recall": 0.98, "precision": 0.41, "f1": 0.58, "auc_roc": 0.69},
        {"modelo": "Regresión logística",
         "recall": 0.94, "precision": 0.68, "f1": 0.79, "auc_roc": 0.90},
        {"modelo": "GradientBoosting + SMOTE (producción)",
         "recall": 0.97, "precision": 0.75, "f1": 0.85, "auc_roc": 0.95},
    ])


# ---------------------------------------------------------------------------
# Escritura a SQLite
# ---------------------------------------------------------------------------
def escribir_sqlite(alumno, riesgo, calificacion, paraescolares, servicio_social, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    con = sqlite3.connect(out_path)
    try:
        alumno.to_sql("alumno", con, index=False)
        riesgo.to_sql("riesgo", con, index=False)
        calificacion.to_sql("calificacion", con, index=False)
        paraescolares.to_sql("paraescolares", con, index=False)
        servicio_social.to_sql("servicio_social", con, index=False)
        usuarios = cargar_usuarios(alumno)
        usuarios.to_sql("usuarios", con, index=False)
        cargar_metricas().to_sql("metricas_modelos", con, index=False)
        con.executescript("""
            CREATE UNIQUE INDEX idx_alumno_nc    ON alumno(no_control);
            CREATE UNIQUE INDEX idx_riesgo_nc    ON riesgo(no_control);
            CREATE UNIQUE INDEX idx_paraesc_nc   ON paraescolares(no_control);
            CREATE UNIQUE INDEX idx_servsoc_nc   ON servicio_social(no_control);
            CREATE UNIQUE INDEX idx_calif_nc_mat ON calificacion(no_control, materia);
            CREATE UNIQUE INDEX idx_usuarios_u   ON usuarios(usuario);
            CREATE TABLE IF NOT EXISTS intervenciones(
                id INTEGER PRIMARY KEY AUTOINCREMENT, no_control TEXT, tipo TEXT,
                comentario TEXT, usuario TEXT, fecha TEXT);
        """)
        con.commit()
    finally:
        con.close()


def reporte(alumno, riesgo, paraescolares, servicio_social):
    def solo_ids(df, ids):
        return int(df["no_control"].isin(set(ids)).sum())
    ids = set(alumno["no_control"])
    print("=" * 60)
    print(f"alumno:        {len(alumno):>5}  alumnos únicos")
    print(f"riesgo:        {len(riesgo):>5}  (features por parcial + baseline)")
    print(f"paraescolares: {len(paraescolares):>5}  "
          f"({solo_ids(paraescolares, ids)} cruzan con alumnos)")
    print(f"servicio_soc.: {len(servicio_social):>5}  "
          f"({solo_ids(servicio_social, ids)} cruzan con alumnos)")
    rojos = int((riesgo["nivel"] == "Rojo").sum())
    amar = int((riesgo["nivel"] == "Amarillo").sum())
    print(f"\nSemáforo baseline (1er parcial): Rojo={rojos}  Amarillo={amar}  "
          f"Verde={len(riesgo) - rojos - amar}")
    n_ss = servicio_social["horas_acumuladas"].fillna(0)
    print(f"Servicio social: {int((n_ss >= HORAS_REQUERIDAS_SS).sum())} de "
          f"{servicio_social['horas_acumuladas'].notna().sum()} con >=480 hrs")


def main():
    ap = argparse.ArgumentParser(description="ETL SARTIA")
    ap.add_argument("--dir", default="BD", help="Carpeta con los archivos fuente (por defecto 'BD', junto a los scripts)")
    ap.add_argument("--out", default="data/sartia.db", help="Ruta de salida SQLite")
    args = ap.parse_args()

    rutas = resolver_rutas(args.dir)
    p_cal = rutas["calificaciones"]
    p_ss = rutas["servicio_social"]
    p_cr = rutas["creditos"]
    print("Archivos detectados:")
    for k, v in rutas.items():
        print(f"  {k:<14} -> {os.path.basename(v)}")

    print("Cargando calificaciones...")
    alumno, riesgo, calif = cargar_calificaciones(p_cal)
    print("Cargando servicio social...")
    ss = cargar_servicio_social(p_ss)
    print("Cargando paraescolares...")
    para = cargar_paraescolares(p_cr)

    escribir_sqlite(alumno, riesgo, calif, para, ss, args.out)
    print(f"\nBase de datos creada: {args.out}")
    reporte(alumno, riesgo, para, ss)
    print("\nCuenta de coordinación: directivo / sartia2026")
    print("Las cuentas de tutor se crean al cargar el primer parcial del ciclo")
    print("en curso (grupos del archivo, contraseñas únicas en data/cuentas_tutores.csv).")
    print("\nSiguiente paso:  python entrenar_modelo.py")


if __name__ == "__main__":
    main()
