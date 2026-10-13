# SARTIA — Tutorial: ETL y conexión del dashboard

Guía paso a paso para integrar las tres fuentes de datos del CBTA 197 en la
base SQLite que consume `app.py` (dashboard de riesgo académico).

---

## 1. Requisitos

- Python 3.10+ con un entorno virtual (ya lo tienes en `.venv`).
- Paquetes: `pandas`, `numpy`, `openpyxl`, `lxml`, `html5lib` (este último
  necesario para leer el `.xls` que en realidad es HTML):

```bash
pip install pandas numpy openpyxl lxml html5lib
```

## 2. Archivos necesarios

Coloca en **una misma carpeta** (p. ej. `codigo_secundario\datos`):

| Archivo | Contenido |
|---|---|
| `detalle_calificaciones ... .xls` | Exportación del sistema escolar (es HTML con extensión .xls). Puede tener cualquier fecha en el nombre. |
| `estatus servicio social feb-jul2026.xlsx` | Hoja **captura** con entregas y horas de servicio social. |
| `creditos formacion academica feb-jul 2026.xlsx` | Hoja **Creditos** con paraescolares. |
| `etl_sartia.py` | Script ETL (entregado en este proyecto). |

> ⚠️ Si tienes **más de un** archivo que empiece con `detalle_calificaciones`,
> el ETL toma el último al ordenar alfabéticamente. Borra o mueve las
> exportaciones viejas para evitar cargar el periodo equivocado.

## 3. Ejecutar el ETL

Desde la terminal, con el `.venv` activado, estando en la carpeta del proyecto:

```bash
python etl_sartia.py --dir "BD" --out data\sartia.db
```

- `--dir` = carpeta donde están los 3 archivos fuente (usa la ruta real, no copies los `...`).
- `--out` = nombre/ruta de la base de datos de salida.

Al arrancar, el script **detecta los archivos automáticamente** por patrón de
nombre y muestra lo que encontró:

```
Archivos detectados:
  calificaciones -> detalle_calificaciones - 2026-10-01T115641.955.xls
  servicio_social -> estatus servicio social feb-jul2026.xlsx
  creditos       -> creditos formacion academica feb-jul 2026.xlsx
```

Salida esperada (con los datos de oct-2026):

```
alumno:         1540  alumnos únicos
riesgo:         1540  (features por parcial + baseline)
paraescolares:  1522  (1521 cruzan con alumnos)
servicio_soc.:  1522  (1521 cruzan con alumnos)

Semáforo baseline (1er parcial): Rojo=114  Amarillo=99  Verde=1327
Servicio social: 475 de 533 con >=480 hrs
```

## 4. Tablas generadas en `sartia.db`

Todas con `no_control` TEXT (14 dígitos) como llave:

- **alumno**: nombre, grupo, generacion, semestre, carrera.
- **riesgo**: materias_p1/p2/p3, reprobadas_p1/p2/p3, promedio_p1/p2/p3,
  asistencia_media_p1/p2/p3, prob_riesgo, nivel
  (Rojo/Amarillo/Verde — baseline de reglas por # de reprobadas en P1;
  la app lo sobrescribe con el modelo al cargar parciales).
- **paraescolares**: tipo_paraescolar, actividad, paraescolar_acreditado_actual,
  paraescolar_total.
- **servicio_social**: **horas_acumuladas** (antes `TOTAL HRS`, causa del error
  original), entregas_realizadas, avance_ss (horas/480), portafolio, carta,
  constancia.

## 5. Conectar `app.py`

Apunta la conexión de tu app a la base generada:

```python
import sqlite3
con = sqlite3.connect("data/sartia.db")   # misma ruta que usa app.py (DB = "data/sartia.db")
```

El query del dashboard ya no falla porque la tabla `servicio_social` ahora sí
tiene la columna `horas_acumuladas`:

```sql
SELECT r.*, a.nombre, a.grupo, a.generacion, a.semestre, a.carrera,
       p.paraescolar_total, s.horas_acumuladas
FROM riesgo r
JOIN alumno a ON a.no_control = r.no_control
LEFT JOIN paraescolares p ON p.no_control = r.no_control
LEFT JOIN servicio_social s ON s.no_control = r.no_control
```

## 6. Limpieza que aplica el ETL (no repitas estos pasos a mano)

1. Filas "NO" (bajas) eliminadas de servicio social.
2. Celdas `#ERROR!` y teléfonos rotos → `NULL` (`pd.to_numeric(errors="coerce")`).
3. Carreras con encoding roto normalizadas.
4. `no_control` unificado a texto de 14 dígitos en las 3 fuentes.
5. Duplicados de servicio social: se conserva el registro con más horas.
6. Horas > 480 **se conservan** tal cual (con `avance_ss` por si el modelo las quiere tratar distinto).

## 7. Solución de problemas

| Error | Causa | Solución |
|---|---|---|
| `FileNotFoundError: ... detalle_calificaciones...` | `--dir` apunta a otra carpeta o copiaste los `...` del ejemplo | Usa la ruta real donde están los archivos |
| `No se encontró el archivo de 'calificaciones'` | El nombre no empieza con `detalle_calificaciones` | Renombra el archivo para que coincida |
| `no such column: s.horas_acumuladas` | Estás usando una base vieja creada a mano | Regenera con `etl_sartia.py` |
| `ImportError` de `read_html` | Falta `lxml`/`html5lib` | `pip install lxml html5lib` |
| `alumno: 0` en el reporte | El `.xls` no es el formato HTML esperado | Verifica que el archivo empiece con `<table` |

## 8. Actualizar datos en un nuevo corte

1. Exporta de nuevo las calificaciones del sistema escolar y guarda el `.xls`
   en la carpeta `datos` (borra el anterior).
2. Actualiza los Excel de servicio social y créditos.
3. Vuelve a correr el comando del paso 3 (sobrescribe `sartia.db`).
4. Reinicia `app.py`.
