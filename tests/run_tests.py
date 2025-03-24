#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import sys
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar tests
from test_data_loader import test_yahoo_source, test_csv_source
from test_rsi_only import test_rsi_calculation as test_rsi_only
from test_simple import test_rsi_calculation as test_rsi_simple, test_csv_path

def run_all_tests():
    """Ejecutar todas las pruebas disponibles."""
    logger.info("Ejecutando todas las pruebas...")
    
    # Lista de todas las funciones de prueba para la Fase 2
    test_functions = [
        ("Test Yahoo Source", test_yahoo_source),
        ("Test CSV Source", test_csv_source),
        ("Test RSI Only", test_rsi_only),
        ("Test RSI Simple", test_rsi_simple),
        ("Test CSV Path", test_csv_path)
        
    ]
    
    # Ejecutar cada prueba y recopilar resultados
    results = []
    for test_name, test_func in test_functions:
        logger.info(f"Ejecutando {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASÓ" if result else "FALLÓ"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} falló con error: {e}")
            results.append((test_name, False))
    
    # Mostrar resumen
    logger.info("\nResumen de pruebas:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    logger.info(f"Pruebas pasadas: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Mostrar detalles de pruebas fallidas
    failed_tests = [(name, result) for name, result in results if not result]
    if failed_tests:
        logger.info("\nPruebas fallidas:")
        for name, _ in failed_tests:
            logger.info(f"- {name}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)