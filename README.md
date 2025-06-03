# Predicción de la severidad de accidentes viales en EE. UU.  
Spark + H2O XGBoost

Proyecto académico que demuestra cómo **replicar de principio a fin** un pipeline distribuido para clasificar la severidad (1–4) de un accidente de tráfico usando PySpark para el preprocesamiento y H2O Flow para entrenar un XGBoost multinomial.

## Dataset

*US-Accidents* (Kaggle) — 1,5 M registros (2016-2023) y 46 columnas tras limpieza. Las variables se agrupan en localización, tiempo, clima, infraestructura vial y la etiqueta `Severity` (1–4).

## Pipeline reproducible

| Paso | Herramienta | Descripción |
|------|-------------|-------------|
| **1. Ingesta** | PySpark | `spark.read.csv` carga el CSV completo en el clúster. |
| **2. Limpieza** | PySpark | – Descarte de columnas con > 50 % nulos.<br>– Imputación de medianas/modas.<br>– Tipado consistente. |
| **3. Feature eng.** | PySpark | • Tokenización + TF sobre `Description`.<br>• One-Hot / Enum encoding en categóricas. |
| **4. EDA** | PySpark + Pandas + Matplotlib | Estadísticas descriptivas, histogramas y heatmaps para detectar outliers y correlaciones clave. |
| **5. Exportación** | PySpark | `df.write.csv()` → `train_ml_ready_unido.csv`, `test_ml_ready_unido.csv`. |
| **6. Entrenamiento** | H2O Flow | Importar CSV, configurar GBM-XGBoost multinomial y entrenar en clúster. |
| **7. Evaluación** | H2O Flow | LogLoss, R², matriz de confusión y variable importance generados automáticamente. |
| **8. Despliegue** | H2O MOJO | Descargar `xgboost-v3.mojo` para inferencia en producción. |

## Parámetros del modelo XGBoost

| Parámetro | Valor |
|-----------|-------|
| `ntrees` | **92** |
| `max_depth` | 20 |
| `min_rows` | 5 |
| `distribution` | multinomial |
| `categorical_encoding` | Enum |
| `calibration_method` | PlattScaling |
| `histogram_type` | UniformAdaptive |
| `seed` | 406 964 802 017 |

## Resultados principales

| Conjunto | LogLoss | RMSE | R² | Hit Rate @1 |
|----------|---------|------|----|-------------|
| **Train** | 0.23499 | 0.2663 | 0.6948 | 0.9057 |
| **Test**  | 0.23462 | 0.2660 | 0.6915 | 0.9057 |

**Matriz de confusión (test)** – las clases 1 y 4 son las más difíciles; clase 2 obtiene recall ≈ 0.99.

## Reproducción paso a paso

1. **Crear entorno**  
   ```bash
   conda create -n accidents python=3.10
   conda activate accidents
   pip install pyspark==3.5.0 h2o==3.46.0.1 jupyterlab matplotlib seaborn
   ```
2. **Clonar y abrir notebook**  
   ```bash
   git clone https://github.com/<tu-usuario>/accident-severity-xgb.git
   cd accident-severity-xgb/notebooks
   jupyter lab proyecto-final-adge-kaggle.ipynb
   ```
3. **Ejecutar todas las celdas** para generar los CSV depurados.
4. **Iniciar H2O Flow**  
   ```bash
   python -m h2o  # abre http://localhost:54321/flow
   ```
5. **En Flow**  
   - *Import Files* → `train_ml_ready_unido.csv` y `test_ml_ready_unido.csv`.  
   - *Build Model* → **Gradient Boosting Machine** y copiar los parámetros de la tabla (sección anterior).  
   - *Predict* sobre test y *Download Model Deployment Package (MOJO)* → `xgboost-v2.mojo`.
6. **Inferencia desde Python**  
   ```python
   import h2o, pandas as pd
   h2o.init()
   mojo = h2o.import_mojo("models/xgboost-v2.mojo")
   new = h2o.H2OFrame(pd.read_csv("data/nuevo_lote.csv"))
   pred = mojo.predict(new).as_data_frame()
   ```

## Hallazgos clave

* **Variables críticas**: visibilidad, condiciones climáticas y presencia de semáforos dominan la importancia relativa.  
* **Desbalance**: la clase 2 representa > 60 % de los registros, lo que explica el recall elevado; futuras versiones deberían aplicar *SMOTE* o *focal loss* para mejorar la clase 4.  
* **Rendimiento**: el MOJO produce inferencias < 10 ms por registro en CPU, apto para sistemas 911 en línea.




## Créditos

Valentina Álvarez, Gabriel Castillo y Daniel Yaruro, Facultad de Ingeniería de Sistemas e Informática (2025).
