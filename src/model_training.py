import pandas as pd
import numpy as np
import argparse
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.label_encoder = None
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        """Entrenar modelo Random Forest"""
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf_classifier.fit(X_train, y_train)
        self.models['random_forest'] = rf_classifier
        return rf_classifier
    
    def train_xgboost(self, X_train, y_train, y_train_encoded, label_encoder, params=None):
        """Entrenar modelo XGBoost"""
        if params is None:
            params = {
                'objective': 'multi:softmax',
                'num_class': len(set(y_train)),
                'max_depth': 6,
                'eta': 0.3,
                'eval_metric': 'merror'
            }
        
        # Crear DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
        
        # Entrenar modelo
        xgb_model = xgb.train(params, dtrain, num_boost_round=100)
        
        self.models['xgboost'] = xgb_model
        self.label_encoder = label_encoder
        return xgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_type='random_forest'):
        """Evaluar modelo y obtener métricas"""
        if model_type == 'random_forest':
            y_pred = model.predict(X_test)
        elif model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_pred_encoded = model.predict(dtest)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded.astype(int))
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return f1, y_pred, report
    
    def save_model(self, model_name, model_path='models/'):
        """Guardar modelo entrenado"""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        if model_name in self.models:
            if model_name == 'xgboost':
                # Guardar modelo XGBoost
                model_file = os.path.join(model_path, 'xgb_model.json')
                self.models[model_name].save_model(model_file)
                # Guardar label encoder
                encoder_file = os.path.join(model_path, 'label_encoder.pkl')
                with open(encoder_file, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                print(f"Modelo XGBoost guardado en {model_file}")
            else:
                # Guardar modelo sklearn (Random Forest)
                model_file = os.path.join(model_path, 'rf_model.pkl')
                with open(model_file, 'wb') as f:
                    pickle.dump(self.models[model_name], f)
                print(f"Modelo {model_name} guardado en {model_file}")

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
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/', 
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--feature_approach',
        type=str,
        default='all_features',
        choices=['all_features', 'reduced_features'],
        help='Feature engineering approach'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='random_forest',
        choices=['random_forest', 'xgboost'],
        help='Type of model to train'
    )
    return parser.parse_args()

def main(input_file, model_file, feature_approach='all_features', model_type='random_forest'):
    print("Cargando datos...")
    df = load_data(input_file)
    
    print("Dividiendo datos...")
    X_train, X_val, y_train, y_val = split_data(df)
    
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
    print(f"Tamaño del conjunto de validación: {X_val.shape}")
    
    # Inicializar trainer
    trainer = ModelTrainer()
    
    print(f"Entrenando modelo {model_type}...")
    
    if model_type == 'random_forest':
        model = trainer.train_random_forest(X_train, y_train)
        f1, y_pred, report = trainer.evaluate_model(model, X_val, y_val, 'random_forest')
        print(f"F1-score en validación: {f1:.4f}")
        trainer.save_model('random_forest', model_file)
    
    elif model_type == 'xgboost':
        # Codificar etiquetas para XGBoost
        y_train_encoded, y_val_encoded, label_encoder = encode_labels(y_train, y_val)
        model = trainer.train_xgboost(X_train, y_train, y_train_encoded, label_encoder)
        f1, y_pred, report = trainer.evaluate_model(model, X_val, y_val, 'xgboost')
        print(f"F1-score en validación: {f1:.4f}")
        trainer.save_model('xgboost', model_file)
    
    print("Entrenamiento completado!")
if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.feature_approach, args.model_type)