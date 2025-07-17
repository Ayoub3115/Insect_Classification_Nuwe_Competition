import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
import argparse
import sys

# PRIMERO configurar el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DESPUÉS importar desde src
from src.data_processing import feature_engineering, preprocess_data

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_type = None
    
    def load_model(self, model_path='models/', model_type='xgboost'):
        """Cargar modelo entrenado"""
        self.model_type = model_type
        
        if model_type == 'xgboost':
            self.model = xgb.Booster()
            model_file = os.path.join(model_path, 'xgb_model.json')
            self.model.load_model(model_file)
            
            # Cargar label encoder
            encoder_file = os.path.join(model_path, 'label_encoder.pkl')
            with open(encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            # Cargar modelo sklearn (Random Forest)
            model_file = os.path.join(model_path, 'rf_model.pkl')
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
    
    def predict(self, X_test):
        """Realizar predicciones"""
        if self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_pred_encoded = self.model.predict(dtest)
            # Para XGBoost, ya obtenemos valores numéricos
            predictions = y_pred_encoded.astype(int)
        else:
            # Para Random Forest, obtenemos nombres de insectos
            predictions_names = self.model.predict(X_test)
            # Convertir nombres a números
            predictions = []
            for pred in predictions_names:
                if pred == 'Aedes':
                    predictions.append(0)
                elif pred == 'Culex':
                    predictions.append(1)
                elif pred == 'Anopheles':
                    predictions.append(2)
                else:
                    predictions.append(0)  # Default
            predictions = np.array(predictions)
        
        return predictions
    
    def predict_test_data(self, test_csv_path='data/test.csv'):
        """Predecir datos de prueba desde CSV"""
        # Cargar datos de prueba
        test_data = pd.read_csv(test_csv_path)
        
        # Preprocesar datos
        test_data_processed = preprocess_data(test_data)
        X_test = feature_engineering(test_data_processed)
        
        # Realizar predicciones
        predictions = self.predict(X_test)
        
        return predictions, test_data
    
    def save_predictions(self, predictions, test_data, output_path='predictions/predictions.json'):
        """Guardar predicciones en formato JSON requerido"""
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Crear diccionario de predicciones usando Unnamed: 0 como clave
        predictions_dict = {}
        for i, pred in enumerate(predictions):
            # Usar la columna Unnamed: 0 como clave
            if 'Unnamed: 0' in test_data.columns:
                key = str(test_data['Unnamed: 0'].iloc[i])
            else:
                key = str(i)
            
            # Asegurar que la predicción es un entero
            predictions_dict[key] = int(pred)
        
        # Crear JSON en el formato requerido con "target"
        final_json = {
            "target": predictions_dict
        }
        
        # Guardar archivo en el formato requerido
        with open(output_path, 'w') as f:
            json.dump(final_json, f, indent=2)
        
        print(f"Predicciones guardadas en {output_path}")
        print(f"Total de predicciones: {len(predictions_dict)}")
        print(f"Formato: {{'target': {{'Unnamed: 0': prediction_value (0, 1, 2)}}}}")
        
        return predictions_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/',
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='xgboost',
        help='Type of model to load (xgboost or random_forest)'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file, model_type='xgboost'):
    try:
        # Usar la clase ModelPredictor para hacer todo el proceso
        predictor = ModelPredictor()
        print(f"Cargando modelo {model_type} desde {model_file}...")
        predictor.load_model(model_file, model_type)
        
        print(f"Prediciendo datos desde {input_file}...")
        predictions, test_data = predictor.predict_test_data(input_file)
        
        print(f"Guardando predicciones en {output_file}...")
        predictor.save_predictions(predictions, test_data, output_file)
        
        print("¡Proceso completado!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file, args.model_type)