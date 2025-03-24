# Tests para el Proyecto de Trading con RL

Esta carpeta contiene las pruebas unitarias y de integración para el proyecto de Trading con Reinforcement Learning.

## Estructura de pruebas

- `test_data_loader.py`: Pruebas para el módulo de carga de datos
- `test_rsi_only.py`: Pruebas específicas para el cálculo del indicador RSI
- `test_simple.py`: Pruebas simples para verificar funcionalidades básicas
- `test_trading_env.py`: Pruebas para el entorno de trading
- `test_dqn_agent.py`: Pruebas para el agente DQN
- `run_tests.py`: Script para ejecutar todas las pruebas

## Cómo ejecutar las pruebas

### Usando run_tests.py

```bash
python tests/run_tests.py
```

### Usando pytest

```bash
pip install pytest  # Si no está instalado
pytest tests/
```

### Ejecutar una prueba específica

```bash
python tests/test_data_loader.py
```

## Añadir nuevas pruebas

Para añadir nuevas pruebas, crea un archivo con el prefijo `test_` y asegúrate de que las funciones de prueba también comiencen con `test_`.