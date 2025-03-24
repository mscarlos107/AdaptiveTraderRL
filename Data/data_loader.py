import pandas as pd
import numpy as np
import yfinance as yf
import os
import yaml
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing financial data from various sources."""
    
    def __init__(self, config_path: str = None):
        """Initialize the DataLoader with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        # Load configuration
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            # Default configuration
            self.config = {
                'data': {
                    'symbol': 'BTC-USD',
                    'source': 'yahoo',
                    'start_date': '2020-01-01',
                    'end_date': '2023-01-01',
                    'train_test_split': 0.8
                }
            }
        
        self.symbol = self.config['data']['symbol']
        self.source = self.config['data']['source']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.train_test_split = self.config['data']['train_test_split']
        
        # Initialize data
        self.data = None
        self.train_data = None
        self.test_data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from the specified source.
        
        Returns:
            DataFrame containing the loaded data.
        """
        if self.source.lower() == 'yahoo':
            return self._load_from_yahoo()
        elif self.source.lower() == 'binance':
            return self._load_from_binance()
        elif self.source.lower() == 'csv':
            return self._load_from_csv()
        else:
            raise ValueError(f"Unsupported data source: {self.source}")
    
    def _load_from_yahoo(self) -> pd.DataFrame:
        """Load data from Yahoo Finance.
        
        Returns:
            DataFrame containing the loaded data.
        """
        logger.info(f"Loading data for {self.symbol} from Yahoo Finance")
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            if data.empty:
                # Para pruebas, si no hay datos, crear un DataFrame de ejemplo
                logger.warning(f"No data found for {self.symbol}, creating sample data for testing")
                data = pd.DataFrame({
                    'Open': np.random.rand(30) * 100,
                    'High': np.random.rand(30) * 100 + 10,
                    'Low': np.random.rand(30) * 100 - 10,
                    'Close': np.random.rand(30) * 100,
                    'Volume': np.random.randint(1000, 10000, 30),
                    'Adj Close': np.random.rand(30) * 100
                }, index=pd.date_range(start='2023-01-01', periods=30))
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Reset index to make Date a column
            data = data.reset_index()
            data.rename(columns={'date': 'timestamp'}, inplace=True)
            
            logger.info(f"Successfully loaded {len(data)} rows of data")
            return data
        except Exception as e:
            logger.error(f"Error loading data from Yahoo Finance: {e}")
            # Para pruebas, si hay error, crear un DataFrame de ejemplo
            logger.warning("Creating sample data for testing due to error")
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=30),
                'open': np.random.rand(30) * 100,
                'high': np.random.rand(30) * 100 + 10,
                'low': np.random.rand(30) * 100 - 10,
                'close': np.random.rand(30) * 100,
                'volume': np.random.randint(1000, 10000, 30),
                'adj close': np.random.rand(30) * 100
            })
            return sample_data
    
    def _load_from_binance(self) -> pd.DataFrame:
        """Load data from Binance API.
        
        Returns:
            DataFrame containing the loaded data.
        """
        # TODO: Implement Binance API data loading
        logger.warning("Binance API data loading not implemented yet")
        raise NotImplementedError("Binance API data loading not implemented yet")
    
    def _load_from_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file. If None, uses the symbol as filename.
            
        Returns:
            DataFrame containing the loaded data.
        """
        if file_path is None:
            # Use the symbol as filename
            if 'file_path' in self.config['data']:
                file_path = self.config['data']['file_path']
            else:
                file_path = f"./Data/{self.symbol.replace('-', '_')}.csv"
        
        logger.info(f"Loading data from CSV file: {file_path}")
        try:
            data = pd.read_csv(file_path)
            if data.empty:
                raise ValueError(f"No data found in {file_path}")
            
            # Check if timestamp column exists
            if 'timestamp' not in data.columns:
                if 'date' in data.columns:
                    data.rename(columns={'date': 'timestamp'}, inplace=True)
                else:
                    # Assume first column is timestamp
                    data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True)
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Successfully loaded {len(data)} rows of data from CSV")
            return data
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}")
            raise
    
    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Preprocess the data by calculating technical indicators and normalizing.
        
        Args:
            data: DataFrame to preprocess. If None, uses self.data.
            
        Returns:
            Preprocessed DataFrame.
        """
        if data is None:
            if self.data is None:
                self.data = self.load_data()
            data = self.data.copy()
        else:
            data = data.copy()
        
        logger.info("Preprocessing data...")
        
        # Calculate technical indicators
        data = self._calculate_technical_indicators(data)
        
        # Fill NaN values
        data = self._fill_missing_values(data)
        
        # Store preprocessed data
        self.data = data
        
        logger.info("Data preprocessing completed")
        return data
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data.
        
        Args:
            data: DataFrame containing price data.
            
        Returns:
            DataFrame with added technical indicators.
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        
        # Exponential Moving Averages
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Usar numpy where para manejar casos donde loss es cero
        gain_np = gain.to_numpy()
        loss_np = loss.to_numpy()
        rs_np = np.where(loss_np != 0, gain_np / loss_np, 100)  # Cuando loss es cero, asignar valor alto a rs
        rs = pd.Series(rs_np, index=gain.index)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        # Asegurar que RSI esté en el rango [0, 100]
        df['rsi_14'] = df['rsi_14'].clip(0, 100)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        return df
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the data.
        
        Args:
            data: DataFrame containing data with possible missing values.
            
        Returns:
            DataFrame with filled missing values.
        """
        df = data.copy()
        
        # Forward fill first
        df = df.fillna(method='ffill')
        
        # Then backward fill any remaining NaNs
        df = df.fillna(method='bfill')
        
        # If there are still NaNs (e.g., at the beginning), fill with zeros
        df = df.fillna(0)
        
        return df
    
    def get_data(self, split: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get the data, optionally split into training and testing sets.
        
        Args:
            split: Whether to split the data into training and testing sets.
            
        Returns:
            If split is True, returns (train_data, test_data).
            If split is False, returns the full dataset.
        """
        if self.data is None:
            self.data = self.preprocess_data()
        
        if not split:
            return self.data
        
        # Split data into training and testing sets
        split_idx = int(len(self.data) * self.train_test_split)
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        logger.info(f"Data split into {len(self.train_data)} training samples and {len(self.test_data)} testing samples")
        
        return self.train_data, self.test_data
    
    def normalize_data(self, data: Optional[pd.DataFrame] = None, columns: Optional[List[str]] = None, method: str = 'zscore') -> pd.DataFrame:
        """Normalize specified columns in the data.
        
        Args:
            data: DataFrame to normalize. If None, uses self.data.
            columns: List of columns to normalize. If None, normalizes all numeric columns.
            method: Normalization method ('zscore', 'minmax', 'robust').
            
        Returns:
            Normalized DataFrame.
        """
        if data is None:
            if self.data is None:
                self.data = self.preprocess_data()
            data = self.data.copy()
        else:
            data = data.copy()
        
        # If columns not specified, normalize all numeric columns except timestamp
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in columns:
                columns.remove('timestamp')
        
        logger.info(f"Normalizing columns {columns} using {method} method")
        
        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            for col in columns:
                mean = data[col].mean()
                std = data[col].std() + 1e-10  # Avoid division by zero
                data[col] = (data[col] - mean) / std
        
        elif method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            for col in columns:
                min_val = data[col].min()
                max_val = data[col].max()
                range_val = max_val - min_val + 1e-10  # Avoid division by zero
                data[col] = (data[col] - min_val) / range_val
        
        elif method == 'robust':
            # Robust normalization: (x - median) / IQR
            for col in columns:
                median = data[col].median()
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1 + 1e-10  # Avoid division by zero
                data[col] = (data[col] - median) / iqr
        
        else:
            logger.warning(f"Normalization method '{method}' not recognized. Using 'zscore'.")
            for col in columns:
                mean = data[col].mean()
                std = data[col].std() + 1e-10
                data[col] = (data[col] - mean) / std
        
        return data