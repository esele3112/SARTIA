# SARTIA — MVP funcional

Sistema de Alerta Temprana de Riesgo Académico y Deserción Escolar basado en Inteligencia Artificial para el CBTA 197.
Entregable 3 — Global Project (20 de octubre de 2026).

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en http://localhost:8501

## Cuentas de prueba (datos ficticios)

| Rol         | Usuario    | Contraseña  | Vista                                        |
|-------------|------------|-------------|----------------------------------------------|
| Coordinación | directivo  | sartia2026  | Todo el plantel + carga de parcial + modelos |
| Tutor (6A)  | tutor.6a   | tutor2026   | Solo grupo 6A, sin carga ni modelos          |

> Las cuentas de tutor del **ciclo en curso** se crean automáticamente al cargar
> el primer parcial (una por grupo, con contraseña única aleatoria en
> `data/cuentas_tutores.csv`). El detalle completo está en
> [MANUAL_DE_OPERACION.md](MANUAL_DE_OPERACION.md).

## Recorrido para la evidencia (video 1–3 min)

1. **P01 Login** — entrar como `directivo`: se ve el Panel general (P02).
2. **P02 Panel general** — KPIs, riesgo por grupo, evidencia de las 3 señales.
3. **P03 Semáforo** — filtrar por nivel Rojo, ordenar por probabilidad.
4. **P04 Ficha de alumno** — explicación del riesgo en lenguaje natural, detalle
   del parcial 1 y registro de intervención (probar guardar sin tipo: muestra error).
5. **Cerrar sesión → entrar como `tutor.6a`** — comprobar que solo ve el grupo 6A
   (restricción aplicada en la consulta, no solo en el menú).
6. **P05 Cargar parcial** (como directivo) — arrastrar un archivo exportado
   (formato del sistema escolar), revisar la vista previa y procesar.
7. **P06 Modelos y métricas** — comparativa vs. la regla del 80 %.
8. **P07 Exportar reporte** — descargar el CSV de alertados.

## Estructura

```
sartia/
├── app.py                     # Aplicación Streamlit (todas las pantallas)
├── etl_sartia.py              # Integra las 3 fuentes del ciclo anterior en SQLite
├── entrenar_modelo.py         # Entrena el modelo con datos reales (inicio de ciclo)
├── generar_datos_demo.py      # Datos sintéticos calibrados (demo sin datos reales)
├── sincronizar_google_sheets.py  # Sync automático de paraescolares y servicio social
├── BD/                        # Los 3 archivos fuente (NO se sube a GitHub: datos del plantel)
├── data/
│   ├── sartia.db              # Base SQLite (alumnos, calificaciones, riesgo, usuarios)
│   ├── sartia_backup.db       # Respaldo manual (no se sube)
│   ├── cuentas_tutores.csv    # Credenciales generadas (NO se sube a GitHub)
│   └── modelo_sartia.pkl      # Modelo entrenado (Gradient Boosting + SMOTE)
├── mockups/                   # Capturas E1–E7 para el documento de evidencias
├── requirements.txt
├── MANUAL_DE_OPERACION.md     # ⭐ Documento principal: operación del ciclo escolar
├── TUTORIAL.md                # Guía del ETL paso a paso
└── README.md
```

## Flujo del ciclo escolar (resumen)

1. **Inicio de ciclo:** ETL (`etl_sartia.py --dir "BD"`) con los archivos del
   ciclo anterior → `entrenar_modelo.py` → levantar la app.
2. **Semanas 6 / 11 / 16:** Cargar parcial 1, 2 y 3 desde la app (directivo).
3. **Cierre de ciclo:** los archivos del ciclo terminado pasan a la carpeta
   `BD` y se repite el paso 1.

El detalle operativo completo está en **MANUAL_DE_OPERACION.md**.

## Declarado como simulado en el MVP

- XGBoost pendiente (el demo usa Gradient Boosting; el cambio es de una línea).
- SHAP formal por alumno pendiente (se muestra importancia global por permutación).
- Sincronización Google Sheets: script listo, no conectado al demo.
- Datos ficticios calibrados a las distribuciones reales del plantel (obligatorio por la guía).
