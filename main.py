import os
import torch
import numpy as np
from data_preprocessing import DataPreprocessor
from pts_model import PTSMLP, PTSTrainer
from trb_model import TRBLSTM, TRBTrainer
from ast_model import ASTAttention, ASTTrainer
from ensemble_model import EnsembleModel

def set_seeds():
    """Establece las semillas para reproducibilidad."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories(output_dir):
    """Crea los directorios necesarios para el proyecto."""
    directories = ['modelos', 'visualizaciones', 'informes']
    for directory in directories:
        os.makedirs(os.path.join(output_dir, directory), exist_ok=True)

def train_individual_models(data_loader, output_dir):
    """Entrena los modelos individuales (PTS, TRB, AST)."""
    # Obtener dimensiones de entrada
    feature_dim = data_loader[0].dataset.get_feature_dim()
    vocab_sizes = data_loader[0].dataset.get_vocab_sizes()
    

    pts_model = PTSMLP(input_size=feature_dim, vocab_sizes=vocab_sizes)
    pts_trainer = PTSTrainer(pts_model, data_loader[0], data_loader[1], data_loader[2], output_dir)
    pts_trainer.train()
    
   
    trb_model = TRBLSTM(input_size=feature_dim, vocab_sizes=vocab_sizes)
    trb_trainer = TRBTrainer(trb_model, data_loader[0], data_loader[1], data_loader[2], output_dir)
    trb_trainer.train()
    
    
    ast_model = ASTAttention(input_size=feature_dim)
    ast_trainer = ASTTrainer(ast_model, data_loader[0], data_loader[1], data_loader[2], output_dir)
    ast_trainer.train()
    
    return pts_model, trb_model, ast_model

def main():
    # Configuración
    data_path = '2024-2025.csv'
    # Asegurar que la ruta de salida sea absoluta
    output_dir = os.path.abspath('nba_predictions')
    print(f"Directorio de salida: {output_dir}")
    
    # Establecer semillas
    set_seeds()
    
    # Crear directorios
    create_directories(output_dir)
    
    # Preprocesar datos
    preprocessor = DataPreprocessor(data_path, output_dir)
    
    # Procesar datos para cada objetivo
    print("Procesando datos para PTS...")
    pts_data = preprocessor.process_data('PTS')
    
    print("Procesando datos para TRB...")
    trb_data = preprocessor.process_data('TRB')
    
    print("Procesando datos para AST...")
    ast_data = preprocessor.process_data('AST')
    
    # Entrenar modelos individuales
    print("\nEntrenando modelo PTS...")
    train_individual_models(pts_data, output_dir)
    
    print("\nEntrenando modelo TRB...")
    train_individual_models(trb_data, output_dir)
    
    print("\nEntrenando modelo AST...")
    train_individual_models(ast_data, output_dir)
    
    # Entrenar modelo de ensamblado
    print("\nEntrenando modelo de ensamblado...")
    ensemble = EnsembleModel(output_dir)
    ensemble.load_pytorch_models()

    # Forzar reentrenamiento de XGBoost (aunque existan .json previos)
    ensemble.train_xgboost_models(
        pts_data[0],   # train_loader
        pts_data[1],   # val_loader
        force_retrain=True
    )
    
    # Evaluar ensamblado
    print("\nEvaluando modelo de ensamblado...")
    results = ensemble.evaluate_ensemble(pts_data[2])  # Usar datos de prueba de PTS como ejemplo
    
    # Guardar modelos de ensamblado
    ensemble.save_models()
    
    print("\n¡Entrenamiento completado!")
    print("\nResultados del ensamblado:")
    for target, metrics in results.items():
        print(f"\n{target.upper()}:")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main() 