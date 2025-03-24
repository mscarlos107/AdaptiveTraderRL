import os
import numpy as np
import pandas as pd
import yaml
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# Importar componentes del proyecto
from Utils.helpers import normalize_data, calculate_sharpe_ratio, plot_equity_curve
from Data.data_loader import DataLoader

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Cargar configuración desde un archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración.
        
    Returns:
        Diccionario con la configuración.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    """Función principal para demostrar el sistema de datos (Fase 2)."""
    # Cargar configuración
    config_path = os.path.join(os.path.dirname(__file__), 'Config', 'config.yaml')
    config = load_config(config_path)
    
    logger.info("Configuración cargada correctamente")
    logger.info(f"Símbolo configurado: {config['data']['symbol']}")
    
    # Crear instancia del DataLoader
    logger.info("Inicializando DataLoader...")
    data_loader = DataLoader(config_path)
    
    # Cargar datos
    logger.info("Cargando datos...")
    data = data_loader.load_data()
    logger.info(f"Datos cargados: {len(data)} filas")
    
    # Preprocesar datos
    logger.info("Preprocesando datos...")
    processed_data = data_loader.preprocess_data(data)
    logger.info(f"Datos preprocesados: {len(processed_data)} filas")
    
    # Dividir en conjuntos de entrenamiento y prueba
    logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    train_data, test_data = data_loader.get_data(split=True)
    logger.info(f"Conjunto de entrenamiento: {len(train_data)} filas")
    logger.info(f"Conjunto de prueba: {len(test_data)} filas")
    
    # Normalizar datos
    logger.info("Normalizando datos...")
    normalized_data = data_loader.normalize_data(train_data)
    logger.info("Datos normalizados correctamente")
    
    # Visualizar algunos datos
    if 'close' in train_data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(train_data['timestamp'], train_data['close'], label='Precio de cierre (train)')
        if 'close' in test_data.columns:
            plt.plot(test_data['timestamp'], test_data['close'], label='Precio de cierre (test)')
        plt.title(f"Datos de {config['data']['symbol']}")
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    logger.info("Demostración del sistema de datos completada")

if __name__ == "__main__":
    main()