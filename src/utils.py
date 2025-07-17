import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

def create_correlation_heatmap(df, figsize=(12, 8), save_path=None):
    """Crear heatmap de correlación"""
    # Limpiar datos y calcular correlación
    df_numeric = df.select_dtypes(include=[np.number])
    df_cleaned = df_numeric.dropna()
    correlation_matrix = df_cleaned.corr()
    
    # Crear figura
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, cbar_kws={"shrink": .8}, linewidths=0.5)
    plt.title('Correlation Matrix of Variables')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_target_distribution(df, target_column='Insect'):
    """Analizar distribución de la variable objetivo"""
    if target_column in df.columns:
        target_counts = df[target_column].value_counts()
        print(f'Distribución de {target_column}:\n{target_counts}')
        return target_counts
    else:
        print(f"Columna {target_column} no encontrada en el DataFrame")
        return None

def check_missing_values(df):
    """Verificar valores faltantes"""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percentage': missing_percentage
    })
    
    return missing_df[missing_df['Missing_Count'] > 0]

def create_directories(base_path='.'):
    """Crear estructura de directorios"""
    directories = [
        'data',
        'src',
        'models',
        'scripts',
        'predictions'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directorio creado: {dir_path}")

def save_example_predictions():
    """Crear archivo de ejemplo de predicciones"""
    example_predictions = {
        "target": {
            "0": 1,
            "1": 2,
            "2": 0,
            "3": 1,
            "4": 2
        }
    }
    
    os.makedirs('predictions', exist_ok=True)
    with open('predictions/example_predictions.json', 'w') as f:
        json.dump(example_predictions, f, indent=2)
    
    print("Archivo de ejemplo creado: predictions/example_predictions.json")

def display_model_comparison(results_dict):
    """Mostrar comparación de modelos"""
    comparison_df = pd.DataFrame(results_dict).T
    print("Comparación de Modelos:")
    print(comparison_df)
    return comparison_df