
---

## Dataset
*US-Accidents* (Kaggle): ~1.5 M registros (2016-2023), 46 columnas, 3.06 GB después de limpieza :contentReference[oaicite:0]{index=0}.  
Variables agrupadas:

| Categoría | Ejemplos |
|-----------|----------|
| Identificación & ubicación | `ID`, `Start_Lat/Lng`, `City`, … |
| Temporales | `Start_Time`, `Sunrise_Sunset`, crepúsculos |
| Meteorológicas | `Temperature`, `Humidity`, `Weather_Condition`, … |
| Infraestructura vial (POI) | `Traffic_Signal`, `Junction`, `Bump`, … |
| **Objetivo** | `Severity` (1-4) |

---

## Pipeline completo
1. **Ingesta distribuida** con PySpark (`spark.read.csv`)  
2. **Curado** : drop de columnas con > 50 % nulos, imputación medianas/modas, tipado consistente:contentReference[oaicite:1]{index=1}  
3. **Ingeniería de texto** sobre `Description` (tokenización + TF)  
4. **EDA visual** (distribución de clases, horas pico, clima, semáforos)  
5. **Export CSV** (`train_ml_ready_unido.csv`, `test_ml_ready_unido.csv`)  
6. **Entrenamiento XGBoost** en H2O Flow (GUI)  
7. **Evaluación & explicación** : métricas, matriz de confusión, variable importance  
8. **Export MOJO** para despliegue

---

## Preprocesamiento y EDA
- **Clase predominante** : Severity 2 > 60 % del total → fuerte desbalance de clases :contentReference[oaicite:2]{index=2}.  
- **Horas críticas** : picos 07 h y 16-17 h (commuting).  
- **Clima** : la mayoría de incidentes ocurre con clima “Fair”.  
- **Semáforos** : ausencia de señal aumenta la frecuencia y la gravedad.

---

## Entrenamiento en H2O Flow
Parámetros finales (selección manual + validación interna) :contentReference[oaicite:3]{index=3}:

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

> **Nota** : La GUI de Flow muestra `xgboost-v3`; el notebook guardó la versión `xgboost-v2` con la misma configuración de fondo.

---

## Resultados
### Métricas globales  
- **Train** : LogLoss 0.23499, RMSE 0.2663, R² 0.6948 :contentReference[oaicite:4]{index=4}  
- **Test** : LogLoss 0.23462, RMSE 0.2660, R² 0.6915 :contentReference[oaicite:5]{index=5}  
- **Hit-rate top-1** : 0.9057 (train) / 0.9057 (test) :contentReference[oaicite:6]{index=6}  

### Matriz de confusión (test)  
| Actual \\ Pred | 1 | 2 | 3 | 4 | Recall |
|---------------|---|---|---|---|--------|
| **1** | 392 | 826 | 15 | 0 | 0.53 |
| **2** | 214 | 109 368 | 4 623 | 71 | 0.99 |
| **3** | 3 | 6 113 | 18 216 | 8 | 0.81 |
| **4** | 0 | 1 531 | 95 | 1 745 | 0.60 | :contentReference[oaicite:7]{index=7}  

### Importancia de variables (top-5) :contentReference[oaicite:8]{index=8}  
1. `feature_98` (22.9 %)  
2. `feature_92`  
3. `feature_108`  
4. `feature_128`  
5. `feature_0` (Speed limit original)  

---

## Reproducibilidad
> **Requisitos**  
> - Python 3.13  
> - Spark 3.5 + Hadoop  
> - H2O ≥ 3.46 (con Flow)  
> - JupyterLab

1. Clonar repo y crear un entorno:
   ```bash
   git clone https://github.com/<tu-usuario>/accident-severity-xgb.git
   cd accident-severity-xgb
   conda env create -f environment.yml   # o pip install -r requirements.txt
   conda activate accidents
