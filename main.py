import os
import numpy as np
import pandas as pd
import yaml
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# Importar componentes del proyecto
from Utils.helpers import normalize_data, calculate_sharpe_ratio, plot_equity_curve

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
    """Función principal para demostrar la infraestructura básica."""
    # Cargar configuración
    config_path = os.path.join(os.path.dirname(__file__), 'Config', 'config.yaml')
    config = load_config(config_path)
    
    logger.info("Configuración cargada correctamente")
    logger.info(f"Símbolo configurado: {config['data']['symbol']}")
    
    # Demostrar funciones de normalización
    data = np.random.randn(100, 5)  # Datos aleatorios para demostración
    normalized_data = normalize_data(data, method='zscore')
    
    logger.info(f"Datos originales: Media={np.mean(data):.4f}, Std={np.std(data):.4f}")
    logger.info(f"Datos normalizados: Media={np.mean(normalized_data):.4f}, Std={np.std(normalized_data):.4f}")
    
    # Demostrar cálculo de métricas financieras
    returns = np.random.normal(0.001, 0.02, 252)  # Retornos aleatorios para demostración
    sharpe = calculate_sharpe_ratio(returns)
    
    logger.info(f"Ratio de Sharpe calculado: {sharpe:.4f}")
    
    # Demostrar visualización
    equity = 10000 * (1 + np.cumsum(returns))
    plot_equity_curve(equity, title="Curva de Equity de Demostración")
    
    logger.info("Demostración de infraestructura básica completada")

if __name__ == "__main__":
    main()