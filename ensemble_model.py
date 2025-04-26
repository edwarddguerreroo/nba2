import xgboost as xgb
import numpy as np
import pandas as pd
import torch
import os
import json
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pts_model import PTSMLP
from trb_model import TRBLSTM
from ast_model import ASTAttention
from xgboost import XGBRegressor
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel:
    def __init__(self, model_dir='nba_predictions'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.preprocessing_info = self._load_preprocessing_info()
        self.categorical_encoders = self.preprocessing_info['categorical_encoders']
        self.scaler_params = self.preprocessing_info['scaler_params']
        
        # Cargar modelos PyTorch
        self.pytorch_models = self.load_pytorch_models()
        
        # Cargar modelos XGBoost
        self.xgb_models = self.load_xgb_models()
        
        print("Modelos cargados exitosamente")
        
    def _load_preprocessing_info(self):
        preprocessing_file = os.path.join(self.model_dir, 'modelos', 'preprocessing_info.json')
        with open(preprocessing_file, 'r') as f:
            return json.load(f)
    
    def load_pytorch_models(self):
        models = {}
        model_files = {
            'PTS': 'pts_mlp.pth',
            'TRB': 'trb_lstm.pth',
            'AST': 'ast_attention.pth'
        }
        
        for target, filename in model_files.items():
            model_path = os.path.join(self.model_dir, 'modelos', filename)
            print(f"Cargando modelo {target} desde {model_path}")
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                model_state = checkpoint['model_state_dict']
                
                # Determinar el tamaño de entrada basado en los pesos guardados
                if 'mlp' in filename:
                    # Para MLP, el tamaño de entrada es el tamaño de input_norm.weight
                    input_size = model_state['input_norm.weight'].shape[0]
                    vocab_sizes = {
                        'Pos': 5,
                        'Team': 30,
                        'Opp': 30,
                        'Pos_Opp': 150
                    }
                    model = PTSMLP(input_size=input_size, vocab_sizes=vocab_sizes, embedding_dim=32)
                elif 'lstm' in filename:
                    # Para LSTM, el tamaño de entrada es el tamaño de input_norm.weight menos el tamaño de los embeddings
                    total_size = model_state['input_norm.weight'].shape[0]
                    embedding_size = 8 * 4  # embedding_dim * número de embeddings
                    input_size = total_size - embedding_size
                    vocab_sizes = {
                        'Pos': 5,
                        'Team': 30,
                        'Opp': 30,
                        'Pos_Opp': 150
                    }
                    model = TRBLSTM(input_size=input_size, vocab_sizes=vocab_sizes, hidden_size=100, num_layers=2, embedding_dim=8)
                else:
                    # Para Attention, el tamaño de entrada es el tamaño de input_norm.weight menos el tamaño de los embeddings
                    total_size = model_state['input_norm.weight'].shape[0]
                    embedding_size = 8 * 4  # embedding_dim * número de embeddings
                    input_size = total_size - embedding_size
                    vocab_sizes = {
                        'Pos': 5,
                        'Team': 30,
                        'Opp': 30,
                        'Pos_Opp': 150
                    }
                    model = ASTAttention(input_size=input_size, vocab_sizes=vocab_sizes, embedding_dim=8)
                
                # Cargar pesos
                model.load_state_dict(model_state)
                model.to(self.device)
                model.eval()
                models[target] = model
                print(f"Modelo {target} cargado exitosamente")
            except Exception as e:
                print(f"Error al cargar el modelo {target}: {str(e)}")
                raise
                
        return models
        
    def load_xgb_models(self):
        models = {}
        for target in ['PTS', 'TRB', 'AST']:
            model_path = os.path.join(self.model_dir, 'modelos', f'ensemble_{target.lower()}_xgb.json')
            if os.path.exists(model_path):
                models[target] = xgb.Booster()
                models[target].load_model(model_path)
                print(f"Modelo XGBoost para {target} cargado exitosamente")
            else:
                print(f"Modelo XGBoost para {target} no encontrado. Se creará uno nuevo durante el entrenamiento.")
                models[target] = None
        return models
        
    def predict(self, features_dict):
        # Preprocesar características
        numeric_features = self._preprocess_numeric_features(features_dict)
        categorical_features = self._encode_categorical_features(features_dict)
        
        # Convertir a tensor y mover al dispositivo correcto
        numeric_tensor = torch.FloatTensor(numeric_features).to(self.device)
        
        # Asegurarse de que el tensor tenga la forma correcta (batch_size=1, seq_len=1, features)
        if len(numeric_tensor.shape) == 1:
            numeric_tensor = numeric_tensor.unsqueeze(0)
        
        predictions = {}
        
        # Predicciones de PyTorch
        for target, model in self.pytorch_models.items():
            with torch.no_grad():
                pred = model(numeric_tensor, torch.tensor([[0, 0, 0, 0]]).to(self.device))
                predictions[f'{target}_torch'] = pred.item()
        
        # Preparar características para XGBoost
        xgb_features = np.concatenate([
            numeric_features,
            [predictions[f'{target}_torch'] for target in ['PTS', 'TRB', 'AST']]
        ])
        
        # Predicciones de XGBoost
        xgb_data = xgb.DMatrix([xgb_features])
        for target, model in self.xgb_models.items():
            predictions[target] = model.predict(xgb_data)[0]
        
        return predictions
        
    def _preprocess_numeric_features(self, features_dict):
        numeric_features = []
        for feature in self.scaler_params['mean_'].keys():
            value = float(features_dict.get(feature, 0.0))
            # Normalizar usando los parámetros del scaler
            normalized_value = (value - self.scaler_params['mean_'][feature]) / self.scaler_params['scale_'][feature]
            numeric_features.append(normalized_value)
        return np.array(numeric_features)
        
    def _encode_categorical_features(self, features_dict):
        categorical_features = []
        for feature, encoder in self.categorical_encoders.items():
            if feature.endswith('_size'):
                continue
            value = features_dict.get(feature, 'OTHER')
            encoded_value = encoder.get(value, len(encoder) - 1)  # Usar el último índice para valores desconocidos
            categorical_features.append(encoded_value)
        return np.array(categorical_features)

    def train_xgboost_models(self, train_loader, val_loader, force_retrain=True):
        """Entrena modelos XGBoost con configuración mejorada y técnicas de reducción de varianza."""
        targets = ['PTS', 'TRB', 'AST']

        for target in targets:
            existing = self.xgb_models.get(target)
            # Entrena solo si no hay modelo o se fuerza retraining
            if existing is None or force_retrain:
                print(f"\nEntrenando modelo XGBoost para {target}")

                # Preparar los datos
                X_train, y_train = self.prepare_xgboost_features(train_loader, target)
                X_val, y_val = self.prepare_xgboost_features(val_loader, target)

                # Inicializar el XGBRegressor
                model = XGBRegressor(
                    n_estimators=1000,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    colsample_bylevel=0.8,
                    colsample_bynode=0.8,
                    gamma=1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    tree_method='hist',
                    grow_policy='lossguide',
                    max_bin=256,
                    max_leaves=0,
                    scale_pos_weight=1,
                    booster='gbtree',
                    objective='reg:squarederror',
                    base_score=0.5,
                    importance_type='gain'
                )

                # Entrenar
                print(f"Iniciando entrenamiento para {target}")
                model.fit(
                    X_train, y_train.ravel(),
                    eval_set=[(X_val, y_val.ravel())],
                    verbose=True,
                )

                # Guardar el modelo entrenado
                self.xgb_models[target] = model
                model_path = os.path.join(self.model_dir, 'modelos', f'ensemble_{target.lower()}_xgb.json')
                model.save_model(model_path)
                print(f"Modelo {target} entrenado y guardado en {model_path}")

            else:
                print(f"Se omite entrenamiento para {target} (modelo existente y force_retrain={force_retrain})")

        return self.xgb_models

    def prepare_xgboost_features(self, data_loader, target_col='pts'):
        """Prepara características para XGBoost con ingeniería de características mejorada."""
        all_features = []
        all_targets = []
        pytorch_preds = {
            'PTS': [],
            'TRB': [],
            'AST': []
        }
        
        # Obtener predicciones de modelos PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for (features, cat_indices), targets in data_loader:
                features = features.to(self.device)
                cat_indices = cat_indices.to(self.device)
                
                # Asegurar que features sea 2D
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                for model_name, model in self.pytorch_models.items():
                    preds = model(features, cat_indices)
                    if preds is not None:
                        preds = preds.cpu().numpy().flatten()
                        pytorch_preds[model_name].extend(preds)
                
                # Guardar características originales y targets
                all_features.append(features.cpu().numpy())
                all_targets.extend(targets.numpy().flatten())  
        
        # Convertir a arrays numpy y asegurar forma 2D
        all_features = np.vstack(all_features)
        all_targets = np.array(all_targets).reshape(-1, 1)  
        
        for k in pytorch_preds:
            pytorch_preds[k] = np.array(pytorch_preds[k]).reshape(-1)
        
        # Crear características adicionales
        enhanced_features = []
        for i in range(len(all_features)):
            row_features = []
            
            # Predicciones de modelos base
            for model_name in ['PTS', 'TRB', 'AST']:
                row_features.append(pytorch_preds[model_name][i])
            
            # Características originales
            row_features.extend(all_features[i].flatten())
            
            # Interacciones entre predicciones
            row_features.append(pytorch_preds['PTS'][i] * pytorch_preds['TRB'][i])  # PTS-TRB
            row_features.append(pytorch_preds['PTS'][i] * pytorch_preds['AST'][i])  # PTS-AST
            row_features.append(pytorch_preds['TRB'][i] * pytorch_preds['AST'][i])  # TRB-AST
            
            # Ratios
            eps = 1e-7
            row_features.append(pytorch_preds['AST'][i] / (pytorch_preds['PTS'][i] + eps))  # AST/PTS
            row_features.append(pytorch_preds['TRB'][i] / (pytorch_preds['PTS'][i] + eps))  # TRB/PTS
            row_features.append(pytorch_preds['AST'][i] / (pytorch_preds['TRB'][i] + eps))  # AST/TRB
            
            # Características cuadráticas
            row_features.append(pytorch_preds['PTS'][i] ** 2)
            row_features.append(pytorch_preds['TRB'][i] ** 2)
            row_features.append(pytorch_preds['AST'][i] ** 2)
            
            enhanced_features.append(row_features)
        
        enhanced_features = np.array(enhanced_features)
        
        # Verificar y limpiar valores no válidos
        enhanced_features = np.nan_to_num(enhanced_features, nan=0.0, posinf=1e6, neginf=-1e6)
        all_targets = np.nan_to_num(all_targets, nan=0.0)
        
        return enhanced_features, all_targets

    def evaluate_ensemble(self, test_loader):
        """Evalúa el modelo de ensamblado con métricas detalladas."""
        X_test, y_test = self.prepare_xgboost_features(test_loader)
        results = {}
        
        for target, model in self.xgb_models.items():
            # Predicciones según tipo de modelo XGBoost
            if isinstance(model, xgb.Booster):
                # Booster acepta DMatrix
                dtest = xgb.DMatrix(X_test)
                y_pred = model.predict(dtest)
            else:
                # XGBRegressor acepta arrays numpy o DataFrame
                y_pred = model.predict(X_test)

            y_true = y_test.ravel()
            
            # Calcular métricas
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # Calcular errores por rango
            ranges = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
            range_metrics = {}
            
            for start, end in ranges:
                mask = (y_true >= start) & (y_true < end)
                if np.any(mask):
                    range_metrics[f'{start}-{end}'] = {
                        'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                        'count': np.sum(mask)
                    }
            
            results[target] = {
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'range_metrics': range_metrics
            }
            
            # Generar visualización mejorada
            self.plot_predictions(y_true, y_pred, target)
            
        # Guardar resultados detallados
        self.save_detailed_report(results)
        
        return results
    
    def plot_predictions(self, y_true, y_pred, target):
        """Genera visualización mejorada de predicciones."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot con transparencia
        plt.scatter(y_true, y_pred, alpha=0.5, label='Predicciones')
        
        # Línea de referencia
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        
        # Agregar bandas de confianza
        sorted_idx = np.argsort(y_true)
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        
        window = len(y_true) // 20
        y_pred_rolling_mean = pd.Series(y_pred_sorted).rolling(window=window, center=True).mean()
        y_pred_rolling_std = pd.Series(y_pred_sorted).rolling(window=window, center=True).std()
        
        plt.fill_between(
            y_true_sorted,
            y_pred_rolling_mean - 2*y_pred_rolling_std,
            y_pred_rolling_mean + 2*y_pred_rolling_std,
            alpha=0.2,
            color='gray',
            label='Intervalo de confianza (95%)'
        )
        
        plt.xlabel(f'{target.upper()} Reales')
        plt.ylabel(f'{target.upper()} Predichos')
        plt.title(f'Predicción de {target.upper()} - Modelo Ensamblado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.savefig(os.path.join(self.model_dir, 'visualizaciones', f'ensemble_{target}_predictions.png'))
        plt.close()
    
    def save_detailed_report(self, results):
        """Guarda un informe detallado de resultados."""
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        for target, metrics in results.items():
            report = {
                'Métricas': {
                    'MAE': convert_to_serializable(metrics['mae']),
                    'R²': convert_to_serializable(metrics['r2']),
                    'RMSE': convert_to_serializable(metrics['rmse'])
                },
                'Métricas por Rango': convert_to_serializable(metrics['range_metrics']),
                'Sugerencias de Mejora': self.generate_improvement_suggestions(metrics)
            }
            
            path = os.path.join(self.model_dir, 'informes', f'ensemble_{target}_report.txt')
            with open(path, 'w') as f:
                json.dump(report, f, indent=4)
    
    def generate_improvement_suggestions(self, metrics):
        """Genera sugerencias de mejora basadas en las métricas."""
        suggestions = []
        
        # Analizar métricas por rango
        for range_name, range_metrics in metrics['range_metrics'].items():
            if range_metrics['mae'] > metrics['mae'] * 1.5:
                suggestions.append(f"Alto error en rango {range_name}. Considerar aumentar muestras en este rango o ajustar el modelo.")
        
        # Analizar R²
        if metrics['r2'] < 0.97:
            suggestions.append("Considerar características adicionales o ajustar hiperparámetros para mejorar R².")
        
        # Analizar RMSE
        if metrics['rmse'] > metrics['mae'] * 1.5:
            suggestions.append("Alta varianza en predicciones. Considerar técnicas de reducción de varianza.")
        
        return suggestions

    def save_models(self):
        """Guarda los modelos XGBoost entrenados."""
        for target, model in self.xgb_models.items():
            path = os.path.join(self.model_dir, 'modelos', f'ensemble_{target.lower()}_xgb.json')
            model.save_model(path)
            print(f"Modelo XGBoost para {target} guardado en {path}")

class MLPModel(nn.Module):
    def __init__(self, input_size, vocab_sizes=None, embedding_dim=32):
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {
                'Pos': 5,
                'Team': 30,
                'Opp': 30,
                'Pos_Opp': 150
            }
        
        # Embeddings para características categóricas
        self.pos_embedding = nn.Embedding(vocab_sizes['Pos'], embedding_dim)
        self.team_embedding = nn.Embedding(vocab_sizes['Team'], embedding_dim)
        self.opp_embedding = nn.Embedding(vocab_sizes['Opp'], embedding_dim)
        self.pos_opp_embedding = nn.Embedding(vocab_sizes['Pos_Opp'], embedding_dim)
        
        # Calcular tamaño total después de concatenar embeddings
        total_embedding_dim = embedding_dim * 4
        total_input_size = input_size + total_embedding_dim
        
        # Normalización de entrada
        self.input_norm = nn.LayerNorm(total_input_size)
        
        # Capas fully connected con batch normalization y layer normalization
        self.fc1 = nn.Linear(total_input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.ln2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.ln3 = nn.LayerNorm(64)
        
        self.output = nn.Linear(64, 1)
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, categorical_indices):
        # Extraer embeddings
        pos_emb = self.pos_embedding(categorical_indices[:, 0])
        team_emb = self.team_embedding(categorical_indices[:, 1])
        opp_emb = self.opp_embedding(categorical_indices[:, 2])
        pos_opp_emb = self.pos_opp_embedding(categorical_indices[:, 3])
        
        # Concatenar embeddings
        embeddings = torch.cat([pos_emb, team_emb, opp_emb, pos_opp_emb], dim=1)
        
        # Concatenar con características numéricas
        x = torch.cat([x, embeddings], dim=1)
        
        # Normalizar entrada
        x = self.input_norm(x)
        
        # Forward pass con skip connections y doble normalización
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.ln1(x1)
        x1 = torch.relu(x1)
        
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.ln2(x2)
        x2 = torch.relu(x2)
        
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = self.ln3(x3)
        x3 = torch.relu(x3)
        
        out = self.output(x3)
        
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, vocab_sizes=None, hidden_size=100, num_layers=2, embedding_dim=8):
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {
                'Pos': 5,
                'Team': 30,
                'Opp': 30,
                'Pos_Opp': 150
            }
        
        # Embeddings para características categóricas
        self.pos_embedding = nn.Embedding(vocab_sizes['Pos'], embedding_dim)
        self.team_embedding = nn.Embedding(vocab_sizes['Team'], embedding_dim)
        self.opp_embedding = nn.Embedding(vocab_sizes['Opp'], embedding_dim)
        self.pos_opp_embedding = nn.Embedding(vocab_sizes['Pos_Opp'], embedding_dim)
        
        # Calcular tamaño total después de concatenar embeddings
        total_embedding_dim = embedding_dim * 4
        total_input_size = input_size + total_embedding_dim
        
        # Normalización de entrada
        self.input_norm = nn.LayerNorm(total_input_size)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=total_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Atención
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def attention_net(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
        
    def forward(self, x, categorical_indices):
        # Extraer embeddings
        pos_emb = self.pos_embedding(categorical_indices[:, 0])
        team_emb = self.team_embedding(categorical_indices[:, 1])
        opp_emb = self.opp_embedding(categorical_indices[:, 2])
        pos_opp_emb = self.pos_opp_embedding(categorical_indices[:, 3])
        
        # Concatenar embeddings
        embeddings = torch.cat([pos_emb, team_emb, opp_emb, pos_opp_emb], dim=1)
        
        # Asegurar que x tenga la forma correcta (batch_size, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Expandir embeddings para que coincida con la longitud de la secuencia
        embeddings = embeddings.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Concatenar con características numéricas y normalizar
        x = torch.cat([x, embeddings], dim=2)
        x = self.input_norm(x)
        
        # Pasar por LSTM
        lstm_out, _ = self.lstm(x)
        
        # Aplicar atención
        context, _ = self.attention_net(lstm_out)
        
        # Predicción final
        out = self.fc(context)
        
        return out

class AttentionModel(nn.Module):
    def __init__(self, input_size, vocab_sizes=None, hidden_size=100, num_layers=2, embedding_dim=8):
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {
                'Pos': 5,
                'Team': 30,
                'Opp': 30,
                'Pos_Opp': 150
            }
        
        # Embeddings para características categóricas
        self.pos_embedding = nn.Embedding(vocab_sizes['Pos'], embedding_dim)
        self.team_embedding = nn.Embedding(vocab_sizes['Team'], embedding_dim)
        self.opp_embedding = nn.Embedding(vocab_sizes['Opp'], embedding_dim)
        self.pos_opp_embedding = nn.Embedding(vocab_sizes['Pos_Opp'], embedding_dim)
        
        # Calcular tamaño total después de concatenar embeddings
        total_embedding_dim = embedding_dim * 4
        total_input_size = input_size + total_embedding_dim
        
        # Normalización de entrada
        self.input_norm = nn.LayerNorm(total_input_size)
        
        # Capa de proyección inicial
        self.projection = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capas de atención
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.2)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_layers)
        ])
        
        # Normalización de capa
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Capa de salida
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x, categorical_indices):
        # Extraer embeddings
        pos_emb = self.pos_embedding(categorical_indices[:, 0])
        team_emb = self.team_embedding(categorical_indices[:, 1])
        opp_emb = self.opp_embedding(categorical_indices[:, 2])
        pos_opp_emb = self.pos_opp_embedding(categorical_indices[:, 3])
        
        # Concatenar embeddings
        embeddings = torch.cat([pos_emb, team_emb, opp_emb, pos_opp_emb], dim=1)
        
        # Asegurar que x tenga la forma correcta (batch_size, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Expandir embeddings para que coincida con la longitud de la secuencia
        embeddings = embeddings.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Concatenar con características numéricas y normalizar
        x = torch.cat([x, embeddings], dim=2)
        x = self.input_norm(x)
        
        # Proyección inicial
        x = self.projection(x)
        
        # Pasar por capas de atención
        for i in range(len(self.attention_layers)):
            # Atención
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms1[i](x + attn_output)
            
            # Feed-forward
            ffn_output = self.ffn_layers[i](x)
            x = self.layer_norms2[i](x + ffn_output)
        
        # Tomar la última posición de la secuencia
        x = x[:, -1, :]
        
        # Predicción final
        out = self.output(x)
        
        return out
    
    def get_pytorch_predictions(self, data_loader):
        """Obtiene predicciones de los modelos PyTorch."""
        predictions = {}
        
        for target, model in self.pytorch_models.items():
            model_preds = []
            with torch.no_grad():
                for batch in data_loader:
                    # Los datos vienen en formato ((features, cat_indices), target)
                    features = batch[0][0].to(self.device)
                    cat_indices = batch[0][1].to(self.device)
                    
                    # Obtener predicción del modelo
                    output = model(features, cat_indices)
                    if output is not None:
                        model_preds.extend(output.cpu().numpy().flatten())
            
            predictions[target] = np.array(model_preds)
        
        return predictions
    
    def save_models(self):
        """Guarda los modelos XGBoost entrenados."""
        for target, model in self.xgb_models.items():
            path = os.path.join(self.model_dir, 'modelos', f'ensemble_{target}_xgb.json')
            model.save_model(path) 