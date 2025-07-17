# 🐞 Insect Classification Challenge

Un sistema de clasificación de insectos basado en lecturas de sensores utilizando Machine Learning. Este proyecto implementa modelos de Random Forest y XGBoost para clasificar insectos en tres categorías: Aedes (0), Culex (1) y Anopheles (2).

## 📁 Estructura del Proyecto

```
nuwe-data-ml1/
├── data/
│   ├── train.csv              # Datos de entrenamiento
│   └── test.csv               # Datos de prueba
├── src/
│   ├── __init__.py           # Módulo Python
│   ├── data_processing.py    # Preprocesamiento de datos
│   ├── model_training.py     # Entrenamiento de modelos
│   └── model_prediction.py   # Predicción y generación de resultados
├── scripts/
│   └── run_pipeline.sh       # Script para ejecutar el pipeline completo
├── models/
│   ├── rf_model.pkl          # Modelo Random Forest entrenado
│   ├── xgb_model.json        # Modelo XGBoost entrenado
│   └── label_encoder.pkl     # Codificador de etiquetas
├── predictions/
│   └── predictions.json      # Archivo de predicciones final
└── README.md
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- Conda o virtualenv

### Instalación de dependencias

```bash
# Crear entorno virtual
conda create -n ml_env python=3.10
conda activate ml_env

# Instalar dependencias
pip install pandas numpy scikit-learn xgboost
```

## 🔧 Uso

### Opción 1: Pipeline Completo (Recomendado)

```bash
# Dar permisos de ejecución al script
chmod +x scripts/run_pipeline.sh

# Ejecutar pipeline completo
./scripts/run_pipeline.sh data/train.csv data/processed_data.csv models/ data/test.csv predictions/predictions.json
```

### Opción 2: Ejecución Manual Paso a Paso

```bash
# 1. Procesamiento de datos
python src/data_processing.py --input_file=data/train.csv --output_file=data/processed_data.csv

# 2. Entrenamiento del modelo (XGBoost)
python src/model_training.py --input_file=data/processed_data.csv --model_file=models/ --model_type=xgboost

# 3. Generación de predicciones
python src/model_prediction.py --input_file=data/test.csv --model_file=models/ --output_file=predictions/predictions.json --model_type=xgboost
```

### Opción 3: Usar Random Forest

```bash
# Cambiar --model_type=xgboost por --model_type=random_forest en los comandos anteriores
python src/model_training.py --input_file=data/processed_data.csv --model_file=models/ --model_type=random_forest
python src/model_prediction.py --input_file=data/test.csv --model_file=models/ --output_file=predictions/predictions.json --model_type=random_forest
```

## 📊 Características del Modelo

### Variables de Entrada
- `Sensor_alpha`: Lectura del sensor alfa
- `Sensor_beta`: Lectura del sensor beta  
- `Sensor_gamma`: Lectura del sensor gamma
- `Sensor_beta_plus`: Lectura adicional del sensor beta
- `Sensor_gamma_plus`: Lectura adicional del sensor gamma
- `Hour`: Hora del día
- `Minutes`: Minutos

### Categorías de Salida
- **0**: Aedes
- **1**: Culex
- **2**: Anopheles

### Algoritmos Implementados

#### 1. XGBoost (Recomendado)
[Es recomendable para un futuro uso hacer GridSearch de los mejores hiperparámetros]
- **Parámetros**: 
  - `objective`: 'multi:softmax'
  - `max_depth`: 6
  - `eta`: 0.3
  - `num_boost_round`: 100
- **Ventajas**: Mayor precisión, manejo automático de valores faltantes

#### 2. Random Forest
- **Parámetros**:
  - `n_estimators`: 100
  - `max_depth`: 10
  - `min_samples_split`: 5
- **Ventajas**: Interpretabilidad, robustez

## 📈 Rendimiento

### Métricas de Evaluación
- **F1-Score**: Métrica principal de evaluación
- **Accuracy**: Precisión general del modelo
- **Classification Report**: Reporte detallado por clase

### Resultados Típicos
- **XGBoost F1-Score**: ~0.87
- **Random Forest F1-Score**: ~0.85

## 📤 Formato de Salida

El archivo `predictions.json` generado tiene el siguiente formato:

```json
{
  "target": {
    "0": 1,
    "1": 0,
    "2": 2,
    "3": 1,
    "4": 2,
    ...
  }
}
```

Donde:
- **Clave**: Índice de la muestra (columna `Unnamed: 0` del test.csv)
- **Valor**: Predicción de categoría (0, 1, o 2)

## 🔧 Configuraciones Avanzadas

### Ingeniería de Características

```bash
# Usar todas las características
--feature_approach=all_features

# Eliminar características correlacionadas
--feature_approach=reduced_features

# Eliminar solo características temporales
--feature_approach=no_time_features
```

### Personalización de Parámetros

Edita los archivos en `src/` para modificar:
- Hiperparámetros de los modelos
- Estrategias de preprocesamiento
- Métricas de evaluación

## 🐛 Resolución de Problemas

### Error: "ModuleNotFoundError: No module named 'src'"
```bash
# Crear archivo __init__.py
touch src/__init__.py
```

### Error: "Permission denied"
```bash
# Dar permisos al script
chmod +x scripts/run_pipeline.sh
```

### Error: "No such file or directory"
```bash
# Verificar estructura de carpetas
mkdir -p data models predictions
```

## 📝 Desarrollo

### Estructura de Clases Principales

1. **ModelTrainer** (`model_training.py`):
   - `train_random_forest()`: Entrena modelo Random Forest
   - `train_xgboost()`: Entrena modelo XGBoost
   - `evaluate_model()`: Evalúa rendimiento
   - `save_model()`: Guarda modelo entrenado

2. **ModelPredictor** (`model_prediction.py`):
   - `load_model()`: Carga modelo entrenado
   - `predict()`: Realiza predicciones
   - `save_predictions()`: Guarda resultados en JSON

### Funciones de Utilidad

- `preprocess_data()`: Limpieza de datos
- `feature_engineering()`: Ingeniería de características
- `split_data()`: División de datos
- `encode_labels()`: Codificación de etiquetas

## 📄 Licencia

Este proyecto fue desarrollado para el desafío de clasificación de insectos de NUWE.

## 👥 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

---

**Desarrollado por**: [Ayoub]  
**Fecha**: Julio 2025  
**Desafío**: NUWE Data Science - Insect Classification
