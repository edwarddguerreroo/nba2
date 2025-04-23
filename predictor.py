import torch
import numpy as np
import pandas as pd
import json
import os
import xgboost as xgb
from ensemble_model import EnsembleModel
from data_preprocessing import DataPreprocessor
import difflib

class NBAPredictor:
    def __init__(self, model_dir='nba_predictions'):
        """
        Inicializa el predictor cargando los modelos y la información de preprocesamiento.
        
        Args:
            model_dir (str): Directorio donde se encuentran los modelos guardados
        """
        self.model_dir = os.path.abspath(model_dir)
        print("Cargando modelos desde:", self.model_dir)
        
        # Cargar información de preprocesamiento
        preprocessing_path = os.path.join(self.model_dir, 'modelos', 'preprocessing_info.json')
        print(f"\nCargando información de preprocesamiento desde: {preprocessing_path}")
        with open(preprocessing_path, 'r') as f:
            self.preprocessing_info = json.load(f)
        
        # Inicializar el modelo de ensamblado
        self.ensemble = EnsembleModel(model_dir=self.model_dir)
        
        # Cargar modelos PyTorch y XGBoost
        print("\nCargando modelos PyTorch...")
        self.ensemble.load_pytorch_models()
        
        # Verificar que los modelos se cargaron correctamente
        print("\nVerificando modelos PyTorch:")
        for name, model in self.ensemble.models.items():
            print(f"\nModelo {name}:")
            print(f"Tipo: {type(model).__name__}")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Número total de parámetros: {total_params:,}")
            print("Estado del modelo:")
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{param_name}: shape={param.shape}, mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        
        # Cargar modelos XGBoost
        print("\nCargando modelos XGBoost...")
        self.ensemble.xgb_models = {}
        for target in ['pts', 'trb', 'ast']:
            model_path = os.path.join(self.model_dir, 'modelos', f'ensemble_{target}_xgb.json')
            print(f"Buscando modelo {target} en: {model_path}")
            if os.path.exists(model_path):
                print(f"Cargando modelo {target}...")
                self.ensemble.xgb_models[target] = xgb.XGBRegressor()
                self.ensemble.xgb_models[target].load_model(model_path)
            else:
                print(f"¡Advertencia! No se encontró el modelo {target}")
        
        # Configurar dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDispositivo: {self.device}")
        
        # Cargar encoders categóricos
        self.categorical_encoders = self.preprocessing_info['categorical_encoders']
        
        # Cargar parámetros del scaler
        self.scaler_mean = np.array(self.preprocessing_info['scaler_params']['mean_'])
        self.scaler_scale = np.array(self.preprocessing_info['scaler_params']['scale_'])
        
        print("\nInicialización completada.")
    
    def _encode_categorical(self, data):
        """Codifica variables categóricas."""
        encoded = {}
        for col in ['Pos', 'Team', 'Opp']:
            # Obtener el encoder para la columna
            encoder = {k: v for k, v in self.categorical_encoders.items() if k.startswith(col + '_')}
            if not encoder:
                continue
            
            # Obtener el valor y codificarlo
            value = data[col]
            encoded[col] = encoder.get(value, len(encoder) - 1)  # Usar último índice si valor no existe
        
        # Manejar Pos_Opp especialmente
        pos = data['Pos']
        opp = data['Opp']
        pos_opp = f"{pos}-{opp}"
        
        # Obtener el encoder para Pos_Opp
        pos_opp_encoder = {k: v for k, v in self.categorical_encoders.items() if k.startswith('Pos_Opp_')}
        if pos_opp_encoder:
            encoded['Pos_Opp'] = pos_opp_encoder.get(pos_opp, len(pos_opp_encoder) - 1)
        else:
            encoded['Pos_Opp'] = 0  # Valor por defecto si no hay encoder
        
        return encoded
    
    def _preprocess_features(self, data):
        """Preprocesa las características para la predicción."""
        # Lista de características numéricas en el orden correcto
        numeric_features = [
            'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
            'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
            'GmSc', 'BPM', '+/-', 'TS%', 'eFG%', 'USG%', 'DRtg', 'ORtg',
            'AST%', 'TRB%', 'STL%', 'BLK%', 'TOV%', 'Home', 'Rest'
        ]
        
        # Extraer características numéricas en el orden correcto
        features = []
        for feature in numeric_features:
            if feature in data:
                try:
                    value = float(data[feature])
                    if np.isnan(value):
                        value = 0.0
                    features.append(value)
                except (ValueError, TypeError):
                    features.append(0.0)
            else:
                features.append(0.0)
        
        # Convertir a array numpy y reshape a 2D
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Normalizar características usando los parámetros del scaler
        features = (features - self.scaler_mean[:features.shape[1]]) / (self.scaler_scale[:features.shape[1]] + 1e-7)
        
        # Convertir a tensor y mover al dispositivo correcto
        features = torch.FloatTensor(features)
        
        # Verificar y limpiar valores no válidos
        features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return features.to(self.device)
    
    def predict(self, player_data):
        """
        Realiza predicciones para un jugador.
        
        Args:
            player_data (dict): Diccionario con los datos del jugador
        
        Returns:
            dict: Predicciones para PTS, TRB y AST
        """
        try:
            # Asegurarse de que tenemos todas las características necesarias
            required_features = ['Pos', 'Team', 'Opp', 'MP']
            for feature in required_features:
                if feature not in player_data:
                    raise ValueError(f"Falta la característica requerida: {feature}")
            
            # Realizar predicción con el ensemble
            predictions = self.ensemble.predict(player_data)
            
            # Redondear predicciones a 1 decimal
            final_predictions = {
                'PTS': round(float(predictions['PTS']), 1),
                'TRB': round(float(predictions['TRB']), 1),
                'AST': round(float(predictions['AST']), 1)
            }
            
            return final_predictions
            
        except Exception as e:
            print(f"Error al realizar predicción: {str(e)}")
            raise
    
    def predict_batch(self, data_list):
        """
        Realiza predicciones para múltiples jugadores.
        
        Args:
            data_list (list): Lista de diccionarios con datos de jugadores
        
        Returns:
            list: Lista de predicciones
        """
        return [self.predict(data) for data in data_list]
    
    def predict_from_csv(self, csv_path):
        """
        Realiza predicciones para todos los jugadores en un archivo CSV.
        
        Args:
            csv_path (str): Ruta al archivo CSV con datos de jugadores
        
        Returns:
            pd.DataFrame: DataFrame con las predicciones
        """
        try:
            # Leer CSV
            df = pd.read_csv(csv_path)
            
            # Realizar predicciones
            predictions = []
            for _, row in df.iterrows():
                pred = self.predict(row.to_dict())
                if pred:
                    predictions.append(pred)
            
            # Convertir a DataFrame
            results = pd.DataFrame(predictions)
            
            # Expandir columna de predicciones
            pred_df = pd.DataFrame(results['predicciones'].tolist())
            results = pd.concat([results.drop('predicciones', axis=1), pred_df], axis=1)
            
            return results
            
        except Exception as e:
            print(f"Error al procesar archivo CSV: {str(e)}")
            return None

def obtener_lista_jugadores():
    """Lee el archivo CSV y devuelve una lista ordenada de nombres de jugadores únicos."""
    try:
        df = pd.read_csv('2024-2025.csv')
        return sorted(df['Player'].unique())
    except Exception as e:
        print(f"Error al leer el archivo de jugadores: {str(e)}")
        return []

def buscar_jugador_similar(nombre, lista_jugadores):
    """Busca nombres similares al proporcionado usando difflib."""
    matches = difflib.get_close_matches(nombre, lista_jugadores, n=1, cutoff=0.6)
    return matches[0] if matches else None

def menu_interactivo():
    """Menú interactivo para realizar predicciones."""
    predictor = NBAPredictor()
    lista_jugadores = obtener_lista_jugadores()
    
    print("=== PREDICTOR NBA ===")
    print("Escribe 'salir' para terminar")
    
    while True:
        nombre = input("\nNombre del jugador: ").strip()
        
        if nombre.lower() == 'salir':
            break
            
        # Buscar jugador similar si no se encuentra exactamente
        if nombre not in lista_jugadores:
            jugador_similar = buscar_jugador_similar(nombre, lista_jugadores)
            if jugador_similar:
                confirmacion = input(f"¿Te refieres a: {jugador_similar}? (s/n): ")
                if confirmacion.lower() != 's':
                    print("Búsqueda cancelada.")
                    continue
                nombre = jugador_similar
            else:
                print("No se encontró ningún jugador similar.")
                continue
        
        try:
            # Leer datos del jugador del CSV
            df = pd.read_csv('2024-2025.csv')
            datos_jugador = df[df['Player'] == nombre].iloc[0].to_dict()
            
            # Preparar datos para la predicción
            datos_prediccion = {
                'Pos': datos_jugador.get('Pos', 'G'),
                'Team': datos_jugador.get('Team', 'LAL'),
                'Opp': datos_jugador.get('Opp', 'GSW'),
                'MP': datos_jugador.get('MP', 30.0),
                'FG': datos_jugador.get('FG', 0.0),
                'FGA': datos_jugador.get('FGA', 0.0),
                'FG%': datos_jugador.get('FG%', 0.0),
                '3P': datos_jugador.get('3P', 0.0),
                '3PA': datos_jugador.get('3PA', 0.0),
                '3P%': datos_jugador.get('3P%', 0.0),
                'FT': datos_jugador.get('FT', 0.0),
                'FTA': datos_jugador.get('FTA', 0.0),
                'FT%': datos_jugador.get('FT%', 0.0),
                'ORB': datos_jugador.get('ORB', 0.0),
                'DRB': datos_jugador.get('DRB', 0.0),
                'TRB': datos_jugador.get('TRB', 0.0),
                'AST': datos_jugador.get('AST', 0.0),
                'STL': datos_jugador.get('STL', 0.0),
                'BLK': datos_jugador.get('BLK', 0.0),
                'TOV': datos_jugador.get('TOV', 0.0),
                'PF': datos_jugador.get('PF', 0.0),
                'PTS': datos_jugador.get('PTS', 0.0)
            }
            
            # Realizar predicción
            predicciones = predictor.predict(datos_prediccion)
            
            # Mostrar resultados
            print(f"\nPredicciones para {nombre}:")
            print(f"PTS: {predicciones['PTS']}")
            print(f"TRB: {predicciones['TRB']}")
            print(f"AST: {predicciones['AST']}")
            
            # Mostrar valores reales si están disponibles
            print("\nValores reales:")
            print(f"PTS: {datos_jugador.get('PTS', 'N/A')}")
            print(f"TRB: {datos_jugador.get('TRB', 'N/A')}")
            print(f"AST: {datos_jugador.get('AST', 'N/A')}")
            
        except Exception as e:
            print(f"Error al procesar el jugador: {str(e)}")

if __name__ == '__main__':
    menu_interactivo() 