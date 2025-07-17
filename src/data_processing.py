import pandas as pd
import numpy as np
import argparse
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """Cargar datos procesados desde archivo CSV"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocesar los datos"""
    df = df.copy()
    # Renombrar columna
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'index'})
    return df

def feature_engineering(df, approach='all_features'):
    """Ingeniería de características"""
    df = df.copy()
    
    # Remover la columna index si existe
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    
    if approach == 'reduced_features':
        # Eliminar variables con alta correlación o poco informativas
        columns_to_drop = ['Sensor_gamma_plus', 'Sensor_beta_plus', 'Minutes', 'Hour', 'Sensor_alpha']
        df = df.drop(columns=columns_to_drop, errors='ignore')
    elif approach == 'no_time_features':
        # Eliminar solo las características de tiempo
        columns_to_drop = ['Minutes', 'Hour']
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Si existe la columna Insect, la removemos para obtener solo features
    if 'Insect' in df.columns:
        X = df.drop(columns=['Insect'])
    else:
        X = df
    
    return X

def split_data(df, test_size=0.2, random_state=42):
    """Dividir datos en entrenamiento y validación"""
    df = preprocess_data(df)
    X = feature_engineering(df)
    y = df['Insect']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def encode_labels(y_train, y_val=None):
    """Codificar las etiquetas para modelos que requieren valores numéricos"""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    if y_val is not None:
        y_val_encoded = label_encoder.transform(y_val)
        return y_train_encoded, y_val_encoded, label_encoder
    
    return y_train_encoded, label_encoder

def train_model(X_train, y_train):
    """Inicializar y entrenar el modelo"""
    # Usando RandomForest como modelo base
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Guardar el modelo entrenado"""
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modelo guardado en: {model_path}")

def get_correlation_matrix(df):
    """Calcular matriz de correlación"""
    df_numeric = df.select_dtypes(include=[np.number])
    correlation_matrix = df_numeric.corr()
    return correlation_matrix

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/train.csv', 
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    parser.add_argument(
        '--feature_approach',
        type=str,
        default='all_features',
        choices=['all_features', 'reduced_features'],
        help='Feature engineering approach'
    )
    return parser.parse_args()

def main(input_file, output_file, feature_approach='all_features'):
    print("Cargando datos...")
    df = load_data(input_file)
    
    print("Preprocesando datos...")
    df_processed = preprocess_data(df)
    
    print("Guardando datos procesados...")
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    print(f"Datos procesados guardados en: {output_file}")
    print("Procesamiento completado!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file, args.feature_approach)