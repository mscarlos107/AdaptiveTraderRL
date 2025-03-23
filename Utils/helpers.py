import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Funciones de normalización
def normalize_data(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """Normaliza los datos utilizando diferentes métodos.
    
    Args:
        data: Array de datos a normalizar.
        method: Método de normalización ('zscore', 'minmax', 'robust').
        
    Returns:
        Array normalizado.
    """
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10  # Evitar división por cero
        return (data - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val + 1e-10  # Evitar división por cero
        return (data - min_val) / range_val
    
    elif method == 'robust':
        # Robust normalization: (x - median) / IQR
        median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1 + 1e-10  # Evitar división por cero
        return (data - median) / iqr
    
    else:
        logger.warning(f"Método de normalización '{method}' no reconocido. Usando 'zscore'.")
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10
        return (data - mean) / std

# Funciones de métricas financieras
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualize: bool = True, periods_per_year: int = 252) -> float:
    """Calcula el Sharpe Ratio de una serie de retornos.
    
    Args:
        returns: Array de retornos porcentuales.
        risk_free_rate: Tasa libre de riesgo anualizada.
        annualize: Si se debe anualizar el resultado.
        periods_per_year: Número de períodos por año (252 días de trading, 12 meses, etc.).
        
    Returns:
        Sharpe Ratio.
    """
    # Convertir tasa libre de riesgo anual a la frecuencia de los retornos
    if annualize:
        rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    else:
        rf_period = risk_free_rate
    
    # Calcular exceso de retorno
    excess_returns = returns - rf_period
    
    # Calcular Sharpe Ratio
    sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
    
    # Anualizar si es necesario
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)
    
    return sharpe

# Funciones de visualización
def plot_equity_curve(equity: np.ndarray, title: str = "Equity Curve") -> None:
    """Grafica la curva de equity.
    
    Args:
        equity: Array con los valores del equity a lo largo del tiempo.
        title: Título del gráfico.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title(title)
    plt.xlabel('Pasos')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()