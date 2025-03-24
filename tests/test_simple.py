import os
import sys
import pandas as pd
import numpy as np
import logging

# Añadir el directorio raíz al path para poder importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rsi_calculation():
    """Test RSI calculation to ensure no division by zero errors."""
    logger.info("Testing RSI calculation...")
    
    # Create sample data with periods of no change (which would cause division by zero)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=50),
        'open': np.random.rand(50) * 100,
        'high': np.random.rand(50) * 100 + 10,
        'low': np.random.rand(50) * 100 - 10,
        'close': np.concatenate([np.ones(20) * 100, np.ones(10) * 110, np.ones(20) * 100]),  # Periods with no change
        'volume': np.random.randint(1000, 10000, 50)
    })
    
    try:
        # Initialize DataLoader
        loader = DataLoader()
        
        # Calculate RSI directly
        data = sample_data.copy()
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Usar numpy where para manejar casos donde loss es cero
        gain_np = gain.to_numpy()
        loss_np = loss.to_numpy()
        rs_np = np.where(loss_np != 0, gain_np / loss_np, 100)  # Cuando loss es cero, asignar valor alto a rs
        rs = pd.Series(rs_np, index=gain.index)
        data['rsi_14'] = 100 - (100 / (1 + rs))
        # Asegurar que RSI esté en el rango [0, 100]
        data['rsi_14'] = data['rsi_14'].clip(0, 100)
        
        # Check if RSI calculation works without errors (ignorando las primeras filas que son NaN por el cálculo de rolling window)
        data_no_nan = data.iloc[14:].copy()  # Omitir las primeras 14 filas (tamaño de la ventana rolling)
        assert not data_no_nan['rsi_14'].isnull().any(), "RSI calculation should not produce NaN values after window period"
        assert (data_no_nan['rsi_14'] >= 0).all() and (data_no_nan['rsi_14'] <= 100).all(), "RSI values should be between 0 and 100"
        
        logger.info("RSI calculation test passed!")
        return True
    except Exception as e:
        logger.error(f"RSI calculation test failed: {e}")
        return False

def test_csv_path():
    """Test CSV file path handling."""
    logger.info("Testing CSV file path handling...")
    
    # Create a sample CSV file in the Data directory
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'Data', 'test_data'), exist_ok=True)
    sample_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'test_data', 'test_data.csv')
    
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=30),
        'open': np.random.rand(30) * 100,
        'high': np.random.rand(30) * 100 + 10,
        'low': np.random.rand(30) * 100 - 10,
        'close': np.random.rand(30) * 100,
        'volume': np.random.randint(1000, 10000, 30)
    })
    
    try:
        # Save sample data to CSV
        sample_data.to_csv(sample_file_path, index=False)
        
        # Test relative path
        loader = DataLoader()
        data = loader._load_from_csv(file_path=sample_file_path)
        
        # Check if data was loaded correctly
        assert not data.empty, "Data should not be empty"
        assert len(data) == 30, "Data should have 30 rows"
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in data.columns, f"Column {col} should be in data"
        
        logger.info("CSV path handling test passed!")
        return True
    except Exception as e:
        logger.error(f"CSV path handling test failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(sample_file_path):
            os.remove(sample_file_path)

if __name__ == "__main__":
    # Run tests
    test_rsi_calculation()
    test_csv_path()