import os
import sys
import pandas as pd
import numpy as np
import yaml
import tempfile
import logging

# Añadir el directorio raíz al path para poder importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yahoo_source():
    """Test loading data from Yahoo Finance."""
    logger.info("Testing Yahoo Finance data source...")
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'source': 'yahoo',
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
            'train_test_split': 0.8
        }
    }
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
        yaml.dump(config, temp)
        temp_path = temp.name
    
    try:
        # Initialize DataLoader with config
        loader = DataLoader(config_path=temp_path)
        
        # Load and preprocess data
        data = loader.load_data()
        assert not data.empty, "Data should not be empty"
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in data.columns, f"Column {col} should be in data"
        
        # Preprocess data
        processed_data = loader.preprocess_data(data)
        assert not processed_data.empty, "Processed data should not be empty"
        
        # Check technical indicators
        indicators = ['sma_10', 'sma_30', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
                     'bb_middle', 'bb_upper', 'bb_lower', 'atr_14']
        for indicator in indicators:
            if indicator in processed_data.columns:
                assert not processed_data[indicator].isnull().all(), f"{indicator} should not be all null"
        
        logger.info("Yahoo Finance data source test passed!")
        return True
    except Exception as e:
        logger.error(f"Yahoo Finance data source test failed: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_csv_source():
    """Test loading data from CSV file."""
    logger.info("Testing CSV data source...")
    
    # Create a sample CSV file
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=30),
        'open': np.random.rand(30) * 100,
        'high': np.random.rand(30) * 100 + 10,
        'low': np.random.rand(30) * 100 - 10,
        'close': np.random.rand(30) * 100,
        'volume': np.random.randint(1000, 10000, 30)
    })
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp:
        sample_data.to_csv(temp.name, index=False)
        temp_path = temp.name
    
    # Create a temporary config file
    config = {
        'data': {
            'symbol': 'test',
            'source': 'csv',
            'file_path': temp_path,
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
            'train_test_split': 0.8
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
        yaml.dump(config, temp_config)
        config_path = temp_config.name
    
    try:
        # Initialize DataLoader with config
        loader = DataLoader(config_path=config_path)
        
        # Load and preprocess data
        data = loader.load_data()
        assert not data.empty, "Data should not be empty"
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in data.columns, f"Column {col} should be in data"
        
        # Preprocess data
        processed_data = loader.preprocess_data(data)
        assert not processed_data.empty, "Processed data should not be empty"
        
        logger.info("CSV data source test passed!")
        return True
    except Exception as e:
        logger.error(f"CSV data source test failed: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    # Run tests
    test_yahoo_source()
    test_csv_source()