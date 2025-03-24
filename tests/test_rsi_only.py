import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rsi_calculation():
    """Test RSI calculation to ensure no division by zero errors."""
    logger.info("Testing RSI calculation...")
    
    # Create sample data with periods of no change (which would cause division by zero)
    sample_data = pd.DataFrame({
        'close': np.concatenate([np.ones(20) * 100, np.ones(10) * 110, np.ones(20) * 100])  # Periods with no change
    })
    
    try:
        # Calculate RSI directly
        data = sample_data.copy()
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Test without protection (should cause division by zero or NaN values)
        try:
            rs_no_protection = gain / loss
            data['rsi_no_protection'] = 100 - (100 / (1 + rs_no_protection))
            if data['rsi_no_protection'].isnull().any():
                logger.info("Division by zero produced NaN values without protection")
            else:
                logger.warning("No NaN values occurred without protection!")
        except Exception as e:
            logger.info(f"Expected error without protection: {e}")
        
        # Test with numpy where protection (should work)
        # Convertir a arrays de numpy para usar np.where
        gain_np = gain.to_numpy()
        loss_np = loss.to_numpy()
        # Cuando loss es cero, asignar un valor muy alto a rs para que RSI sea exactamente 100
        rs_np = np.where(loss_np != 0, gain_np / (loss_np + 1e-10), 1000000)  # Valor muy alto para que RSI sea prácticamente 100
        # Convertir de vuelta a serie de pandas
        rs = pd.Series(rs_np, index=gain.index)
        data['rsi_protected'] = 100 - (100 / (1 + rs))
        # Asegurar que RSI esté en el rango [0, 100]
        data['rsi_protected'] = data['rsi_protected'].clip(0, 100)
        
        # Verificar que no hay valores NaN (ignorando las primeras filas que son NaN por el cálculo de rolling window)
        data_no_nan = data.iloc[14:].copy()  # Omitir las primeras 14 filas (tamaño de la ventana rolling)
        assert not data_no_nan['rsi_protected'].isnull().any(), "RSI calculation should not produce NaN values after window period"
        
        # Verificar que los valores están en el rango [0, 100] (ignorando las primeras filas que son NaN)
        assert (data_no_nan['rsi_protected'] >= 0).all() and (data_no_nan['rsi_protected'] <= 100).all(), "RSI values should be between 0 and 100"
        
        # Verificar específicamente los valores cuando loss es cero
        zero_loss_indices = loss.index[loss == 0]
        if not zero_loss_indices.empty:
            for idx in zero_loss_indices:
                if idx in data.index:
                    # Verificar que el RSI es aproximadamente 100 cuando loss es 0 (con una pequeña tolerancia)
                    assert abs(data.loc[idx, 'rsi_protected'] - 100) < 0.01, f"RSI should be approximately 100 when loss is 0, got {data.loc[idx, 'rsi_protected']} at index {idx}"
        
        logger.info("RSI calculation test passed!")
        return True
    except Exception as e:
        logger.error(f"RSI calculation test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    test_rsi_calculation()