# SARTIA — Manual de Operación

Sistema de Alerta Temprana de Riesgo Académico · CBTA 197
Ciclo escolar sept–dic 2026 (y siguientes)

---

## 1. Filosofía del sistema

El modelo se **entrena una vez al inicio de cada ciclo** con los datos
**completos del ciclo anterior** (parciales 1–3 + calificaciones finales).
La calificación final sirve de *etiqueta*: sabemos quién reprobó.

Durante el ciclo en curso **no se reentrena**: solo se cargan los parciales
1, 2 y 3 a medida que el sistema escolar los libera, y el modelo entrenado
produce las predicciones del semáforo.

```
Ciclo anterior (feb–jul)          Ciclo actual (sept–dic)
┌─────────────────────────┐       ┌──────────────────────────────────┐
│ ETL (3 archivos)        │       │ Semana 6:  Cargar parcial 1      │
│ entrenar_modelo.py      │─────▶ │ Semana 11: Cargar parcial 2      │
│ modelo + base listos    │       │ Semana 16: Cargar parcial 3      │
└─────────────────────────┘       │ Semáforo actualizado cada carga  │
                                  └──────────────────────────────────┘
                                            │
                                            ▼
                              Fin de ciclo: los archivos de sept–dic
                              se convierten en el "ciclo anterior"
                              del siguiente entrenamiento.
```

---

## 2. Requisitos previos

- Python 3.12 con virtualenv en `.venv` (ya configurado).
- Dependencias: `pip install -r requirements.txt`.
- Los 3 archivos fuente del **ciclo anterior** en la carpeta **`BD`**
  (dentro del proyecto, junto a los scripts):
  | Archivo (empieza con…) | Contenido |
  |---|---|
  | `detalle_calificaciones…` | HTML del sistema escolar: parciales 1–3 + final |
  | `estatus servicio social…` | Excel, hoja `captura` |
  | `creditos formacion academica…` | Excel, hoja `Creditos` |

---

## 3. Puesta en marcha (una vez por ciclo, ~10 minutos)

> ⚠️ **Streamlit debe estar DETENIDO** durante estos pasos (`Ctrl+C` en su
> terminal, o `taskkill /F /IM python.exe` si quedó huérfano), porque la
> base y el modelo están bloqueados mientras la app corre.

```bat
:: 0. Activar entorno (desde la carpeta del proyecto)
cd "ruta\del\proyecto\sartia"
.venv\Scripts\activate

:: 1. Respaldo de seguridad (recomendado)
copy data\sartia.db data\sartia_backup.db

:: 2. ETL: construye la base con el ciclo anterior (historial + entrenamiento)
python etl_sartia.py --dir "BD" --out data\sartia.db

:: 3. Entrenar el modelo con datos reales
python entrenar_modelo.py

:: 4. Levantar la aplicación
streamlit run app.py
```

Verifica en la consola del paso 2 que los 3 archivos fueron detectados y en
el paso 3 que las métricas del Gradient Boosting se imprimen sin errores.

---

## 4. Operación semanal (cargar parciales)

1. Exportar del sistema escolar el detalle de calificaciones del parcial
   correspondiente (mismo formato `.xls` de siempre).
2. Entrar a SARTIA como **directivo** → menú **Cargar parcial**.
3. Seleccionar el número de parcial (1 ≈ semana 6, 2 ≈ semana 11,
   3 ≈ semana 16).
4. Arrastrar el archivo y revisar la **vista previa**:
   registros leídos y alumnos reconocidos en el padrón.
5. Clic en **Procesar y recalcular predicciones**.
6. Revisar el **Semáforo de riesgo** y atender a los alumnos en rojo:
   registrar la intervención en su **Ficha de alumno**.

> Las features del modelo son **acumulativas**: al cargar el parcial 2 se
> usan los promedios de parciales 1+2, y al cargar el 3, los de 1+2+3.
> No es necesario volver a subir parciales anteriores: cada exportación del
> sistema escolar ya contiene las columnas de los parciales pasados.

---

## 5. Cierre de ciclo (repetir sección 3)

Cuando el ciclo termine y el sistema escolar tenga las calificaciones
finales, copia los 3 archivos a tu carpeta de datos y repite los pasos de
la sección 3. El nuevo modelo se entrena con el ciclo que acaba de cerrarse
y el historial de la base se renueva.

---

## 6. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `PermissionError: [WinError 32]` al correr ETL | Streamlit tiene abierta la base | Detener Streamlit (`Ctrl+C` / `taskkill /F /IM python.exe`) |
| `No module named '_loss'` al cargar parcial | El `.pkl` fue entrenado con otra versión de sklearn | Correr `python entrenar_modelo.py` en este entorno |
| `TypeError: dtype 'str' does not support operation 'mean'` | Columnas del Excel llegan como texto | Ya corregido en `app.py` (conversión forzada a numérico) |
| `no such table: intervenciones` | Base creada antes de que existiera la tabla | Ya corregido: la app la crea al arrancar y el ETL también |
| `KeyError: 'para_pts'` (o similar) al predecir | Nombres de columnas del modelo ≠ del dataframe | Ya corregido: renombrado automático en `app.py` |
| "Números de control no encontrados" en vista previa | Alumnos de nuevo ingreso no están en el padrón | Normal al inicio de ciclo; se incorporan al ETL del siguiente ciclo |
| `WinError 3` al correr el ETL | La ruta `--dir` no existe | Verificar la ruta entre comillas |

---

## 7. Estructura de archivos

```
codigo_secundario/
├── app.py                  # Dashboard Streamlit (todas las pantallas)
├── etl_sartia.py           # Integra las 3 fuentes del ciclo anterior en SQLite
├── entrenar_modelo.py      # Entrena el modelo con los datos reales (nuevo)
├── generar_datos_demo.py   # Datos sintéticos + modelo demo (solo para demo sin datos reales)
├── sincronizar_google_sheets.py
├── data/
│   ├── sartia.db           # Base del ciclo (NO editar a mano)
│   ├── sartia_backup.db    # Respaldo manual
│   └── modelo_sartia.pkl   # Modelo de producción
├── requirements.txt
└── MANUAL_DE_OPERACION.md
```

---

## 8. Cuentas de acceso

| Rol | Usuario | Contraseña | Permisos |
|---|---|---|---|
| Coordinación | `directivo` | `sartia2026` | Todo + carga de parciales |
| Tutor de cada grupo | `tutor.<grupo>` | **única, aleatoria** | Solo su grupo (sin carga) |

**Ciclo de vida de las cuentas de tutor:**

1. **ETL (inicio de ciclo):** crea únicamente la cuenta `directivo` y
   **reinicia** `data/cuentas_tutores.csv` (ciclo nuevo = credenciales
   nuevas). No crea cuentas de tutor.
2. **Al subir el primer parcial del ciclo en curso:** la app crea las cuentas
   de tutor con los grupos **del archivo** (p. ej. sept–dic: semestres 1, 3, 5):
   - una cuenta por grupo (`tutor.1a`, `tutor.3a-tp`, `tutor.5a-tpia`…), cada
     una con **su propia contraseña aleatoria** (se agregan al CSV);
   - si se vuelve a cargar un parcial y los grupos cambiaron, elimina las
     cuentas de grupos que ya no existen en el ciclo.
   - la app muestra un aviso listando las cuentas recién creadas.
3. **Distribución:** entrega a cada tutor su usuario y contraseña de forma
   privada y **borra `data/cuentas_tutores.csv`** después.

Ejemplo de cuentas tras subir el parcial 1 de sept–dic:

| Grupo | Usuario |
|---|---|
| 1A | `tutor.1a` |
| 3A TP | `tutor.3a-tp` |
| 5A TPIA | `tutor.5a-tpia` |

> Cambiar las contraseñas del directivo y de todos los tutores antes de
> cualquier uso fuera de la evidencia académica.
