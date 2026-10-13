#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARTIA - Sincronizador de fuentes de Google Sheets
==================================================
Jala de forma automatica las hojas de:
  1) Creditos de formacion academica (paraescolares)
  2) Estatus de servicio social

y las carga en sartia.db (SQLite). La unica fuente manual sigue siendo
el archivo de calificaciones por parcial.

Uso:
  python sincronizar_google_sheets.py                # sincroniza todo
  python sincronizar_google_sheets.py --check-only   # valida sin escribir
  python sincronizar_google_sheets.py --solo creditos

Requisitos:
  pip install gspread pandas openpyxl python-dotenv

Configuracion (variables de entorno o archivo .env):
  GOOGLE_CREDENTIALS=ruta/a/credenciales.json   (cuenta de servicio, SOLO LECTURA)
  SARTIA_DB=data/sartia.db
  ALERTA_EMAIL=correo@cbta197.net   (opcional; vacio = solo log)
  SMTP_SERVER / SMTP_USER / SMTP_PASS  (opcional)
"""

import argparse
import datetime as dt
import json
import logging
import os
import re
import sqlite3
import sys
import unicodedata

import pandas as pd

try:
    import gspread
except ImportError:
    sys.exit("Falta gspread: pip install gspread")

# ----------------------------------------------------------------------------
# CONFIGURACION DE FUENTES
# Aqui se registran las hojas. Agregar una fuente nueva = agregar un dict.
# 'columnas' define el esquema esperado: nombre_hoja -> (nombre_normalizado, tipo)
# ----------------------------------------------------------------------------
FUENTES = [
    {
        "id": "creditos",
        "sheet_key": os.getenv("SHEET_CREDITOS_KEY", "PEGAR_ID_DE_LA_HOJA_DE_CREDITOS"),
        "worksheet": "Creditos",
        "tabla_destino": "paraescolares",
        "requeridas": ["no_control", "generacion", "grupo", "paraescolar_total"],
        "columnas": {
            "No. Control": ("no_control", "texto"),
            "GENERACION": ("generacion", "texto"),
            "GRUPO": ("grupo", "texto"),
            "APELLIDO PATERNO": ("ap_paterno", "texto"),
            "APELLIDO MATERNO": ("ap_materno", "texto"),
            "NOMBRE": ("nombre", "texto"),
            "CURP": ("curp", "texto"),
            "PARAESCOLAR TOTAL": ("paraescolar_total", "numero"),
            "CREDITOS TOTALES": ("creditos_totales", "numero"),
            "TOTALES": ("totales", "numero"),
            "PARA. FEB-JUL 2026": ("para_sem_actual", "numero"),
        },
        "limpiar": "creditos",
    },
    {
        "id": "servicio_social",
        "sheet_key": os.getenv("SHEET_SS_KEY", "PEGAR_ID_DE_LA_HOJA_DE_SERVICIO_SOCIAL"),
        "worksheet": "captura",
        "tabla_destino": "servicio_social",
        "requeridas": ["no_control", "horas_acumuladas"],
        "columnas": {
            "control": ("no_control", "texto"),
            "NOMBRE": ("nombre", "texto"),
            "GENERACION": ("generacion", "texto"),
            "GRUPO": ("grupo", "texto"),
            "CURP": ("curp", "texto"),
            "GENERO": ("genero", "texto"),
            "ALTA": ("alta", "texto"),
            "LUGAR": ("lugar", "texto"),
            "TOTAL HRS": ("horas_acumuladas", "numero"),
            "PORTAFOLIO": ("portafolio", "texto"),
            "CARTA": ("carta", "texto"),
            "CONSTANCIA": ("constancia", "texto"),
            "16. PROG": ("programa", "texto"),
        },
        "limpiar": "servicio_social",
    },
]

DB_PATH = os.getenv("SARTIA_DB", "data/sartia.db")
LOG_DIR = "logs"

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "sync_google_sheets.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sartia.sync")


# ----------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------
def quitar_invisibles(valor):
    """Quita caracteres de control Unicode invisibles que rompen Excel/Sheets."""
    if not isinstance(valor, str):
        return valor
    return re.sub(r"[\u202a-\u202e\u200e\u200f\ufeff]", "", valor).strip()


def normalizar_columna(col):
    col = unicodedata.normalize("NFKD", str(col)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")


def a_numero(serie):
    """Coerce a numero; valores basura ('NO', '', '#ERROR!') -> NaN."""
    return pd.to_numeric(
        serie.astype(str)
        .map(quitar_invisibles)
        .str.replace(r"\.0$", "", regex=True)
        .replace({"": None, "NO": None, "nan": None, "None": None}),
        errors="coerce",
    )


def limpiar_df(df, fuente_id):
    """Reglas de limpieza especificas por fuente (lo detectado en ETL exploratorio)."""
    if fuente_id == "servicio_social":
        # La hoja de SS tiene el encabezado duplicado como primera fila de datos
        col0 = df.columns[0]
        df = df[df[col0].astype(str).str.strip().str.lower() != col0.lower()].copy()
        # Recalcular horas si el total viene vacio pero hay entregas
        hrs_cols = [c for c in df.columns if re.fullmatch(r"h\.?\d*|hrs\d*", c)]
    if fuente_id == "creditos":
        # A veces el acreditado viene como True/'Acreditado'/'NO TRUE' -> normalizar
        for c in df.columns:
            if "acredit" in c:
                df[c] = (
                    df[c].astype(str).str.strip().str.upper()
                    .map(lambda x: "ACREDITADO" if x in ("TRUE", "ACREDITADO", "SI", "SÍ") else "NO ACREDITADO")
                )
    return df


# ----------------------------------------------------------------------------
# VALIDACION DE ESQUEMA
# ----------------------------------------------------------------------------
def validar_esquema(df, fuente):
    """Compara columnas recibidas vs esperadas. Devuelve (df_ok, problemas)."""
    problemas = []
    faltantes = [c for c in fuente["columnas"] if c not in df.columns]
    if faltantes:
        # Busqueda tolerante: quizas renombraron quitando acentos/puntos
        mapeo_tol = {}
        cols_norm = {normalizar_columna(c): c for c in df.columns}
        for f in faltantes:
            clave = normalizar_columna(f)
            if clave in cols_norm:
                mapeo_tol[f] = cols_norm[clave]
            else:
                problemas.append(f"FALTA COLUMNA REQUERIDA: '{f}'")
        df = df.rename(columns=mapeo_tol)

    for col, (_, tipo) in fuente["columnas"].items():
        if col in df.columns and tipo == "numero":
            antes = df[col].notna().mean()
            df[col] = a_numero(df[col])
            if df[col].notna().mean() < antes - 0.10:  # perdimos >10% de datos
                problemas.append(f"AVISO: columna '{col}' perdio muchos valores al normalizar")

    req = [r for r in fuente["requeridas"] if r in [v[0] for v in fuente["columnas"].values()]]
    for r in req:
        col_orig = next(k for k, v in fuente["columnas"].items() if v[0] == r)
        if col_orig in df.columns and df[col_orig].isna().all():
            problemas.append(f"CRITICO: columna clave '{col_orig}' esta 100% vacia")

    return df, problemas


# ----------------------------------------------------------------------------
# CARGA A SQLITE (upsert simple por no_control)
# ----------------------------------------------------------------------------
DDL = {
    "paraescolares": """
        CREATE TABLE IF NOT EXISTS paraescolares (
            no_control TEXT PRIMARY KEY,
            generacion TEXT, grupo TEXT,
            ap_paterno TEXT, ap_materno TEXT, nombre TEXT, curp TEXT,
            paraescolar_total REAL, creditos_totales REAL, totales REAL,
            para_sem_actual REAL, ultima_sync TEXT
        )""",
    "servicio_social": """
        CREATE TABLE IF NOT EXISTS servicio_social (
            no_control TEXT PRIMARY KEY,
            nombre TEXT, generacion TEXT, grupo TEXT, curp TEXT, genero TEXT,
            alta TEXT, lugar TEXT, programa TEXT,
            horas_acumuladas REAL, portafolio TEXT, carta TEXT, constancia TEXT,
            ultima_sync TEXT
        )""",
}


def upsert(df, tabla, conn):
    ahora = dt.datetime.now().isoformat(timespec="seconds")
    df["ultima_sync"] = ahora
    cols = list(df.columns)
    marcadores = ",".join("?" * len(cols))
    update = ",".join(f"{c}=excluded.{c}" for c in cols if c != "no_control")
    sql = (f"INSERT INTO {tabla} ({','.join(cols)}) VALUES ({marcadores}) "
           f"ON CONFLICT(no_control) DO UPDATE SET {update}")
    conn.executemany(sql, df.itertuples(index=False, name=None))
    conn.commit()


# ----------------------------------------------------------------------------
# ALERTA (correo opcional; si no hay SMTP, solo log fuerte)
# ----------------------------------------------------------------------------
def alertar(asunto, cuerpo):
    log.error("ALERTA: %s | %s", asunto, cuerpo)
    destinatario = os.getenv("ALERTA_EMAIL")
    if not destinatario:
        return
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = f"[SARTIA] {asunto}"
        msg["From"] = os.getenv("SMTP_USER")
        msg["To"] = destinatario
        msg.set_content(cuerpo)
        with smtplib.SMTP(os.getenv("SMTP_SERVER", "smtp.gmail.com"), 587) as s:
            s.starttls()
            s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            s.send_message(msg)
        log.info("Alerta enviada a %s", destinatario)
    except Exception as e:  # nunca romper el sync por un correo
        log.error("No se pudo enviar correo de alerta: %s", e)


# ----------------------------------------------------------------------------
# FLUJO PRINCIPAL
# ----------------------------------------------------------------------------
def sincronizar(fuente, gc, check_only=False):
    log.info("--- Sincronizando fuente '%s' (hoja '%s') ---", fuente["id"], fuente["worksheet"])
    try:
        sh = gc.open_by_key(fuente["sheet_key"])
        ws = sh.worksheet(fuente["worksheet"])
        datos = ws.get_all_records()
        df = pd.DataFrame(datos)
        log.info("Registros leidos: %d", len(df))
    except gspread.exceptions.APIError as e:
        alertar(f"Acceso denegado a '{fuente['id']}'", str(e))
        return False
    except Exception as e:
        alertar(f"Error leyendo '{fuente['id']}'", repr(e))
        return False

    df = limpiar_df(df, fuente["id"])
    df, problemas = validar_esquema(df, fuente)
    if problemas:
        for p in problemas:
            (alertar if p.startswith("CRITICO") or p.startswith("FALTA") else log.warning)(
                f"Esquema '{fuente['id']}'", p
            )

    # Renombrar a nombres normalizados y quedarnos solo con las columnas del config
    renombrar = {k: v[0] for k, v in fuente["columnas"].items() if k in df.columns}
    df = df.rename(columns=renombrar)[[c for c in renombrar.values()]]
    df["no_control"] = df["no_control"].astype(str).map(quitar_invisibles).str.replace(r"\.0$", "", regex=True)
    df = df[df["no_control"].notna() & (df["no_control"].str.len() >= 6)]
    df = df.drop_duplicates("no_control", keep="last")
    log.info("Registros validos tras limpieza: %d", len(df))

    if check_only:
        log.info("[CHECK-ONLY] No se escribio a la base de datos.")
        return True

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(DDL[fuente["tabla_destino"]])
    upsert(df, fuente["tabla_destino"], conn)
    conn.close()
    log.info("Cargados %d registros en tabla '%s'", len(df), fuente["tabla_destino"])
    return True


def main():
    parser = argparse.ArgumentParser(description="SARTIA - Sync Google Sheets")
    parser.add_argument("--check-only", action="store_true", help="Valida sin escribir")
    parser.add_argument("--solo", help="IDs de fuentes separados por coma (ej: creditos)")
    args = parser.parse_args()

    cred_path = os.getenv("GOOGLE_CREDENTIALS")
    if not cred_path or not os.path.exists(cred_path):
        sys.exit("Defina GOOGLE_CREDENTIALS con la ruta al JSON de la cuenta de servicio.")

    with open(cred_path) as f:  # nunca loguear el contenido de las credenciales
        json.load(f)

    gc = gspread.service_account(filename=cred_path)
    seleccion = [s.strip() for s in args.solo.split(",")] if args.solo else None
    resumen = []
    for fuente in FUENTES:
        if seleccion and fuente["id"] not in seleccion:
            continue
        ok = sincronizar(fuente, gc, check_only=args.check_only)
        resumen.append((fuente["id"], "OK" if ok else "ERROR"))

    log.info("=== RESUMEN: %s ===", ", ".join(f"{k}={v}" for k, v in resumen))
    if any(v == "ERROR" for _, v in resumen):
        sys.exit(1)


if __name__ == "__main__":
    main()
