# Trading Adaptativo con Aprendizaje por Refuerzo - Fase 2: Sistema de Datos

## Descripción General del Proyecto

Este proyecto implementa un sistema de trading adaptativo utilizando técnicas de Aprendizaje por Refuerzo (Reinforcement Learning o RL). El objetivo principal es desarrollar agentes inteligentes capaces de tomar decisiones de compra y venta en mercados financieros, adaptándose a las condiciones cambiantes del mercado y optimizando el rendimiento de la cartera.

### Motivación

Los mercados financieros son entornos complejos, dinámicos y con alta incertidumbre, lo que los convierte en un campo ideal para la aplicación de técnicas de aprendizaje por refuerzo. A diferencia de los enfoques tradicionales basados en reglas predefinidas o análisis estadístico, los agentes de RL pueden:

- Aprender de forma autónoma a partir de la experiencia
- Adaptarse a cambios en las condiciones del mercado
- Optimizar objetivos a largo plazo (como el ratio de Sharpe) en lugar de simplemente maximizar los retornos a corto plazo
- Gestionar el equilibrio entre exploración de nuevas estrategias y explotación de estrategias conocidas

### Enfoque Técnico

El sistema se basa en la arquitectura de aprendizaje por refuerzo profundo (Deep Reinforcement Learning), utilizando redes neuronales para aproximar funciones de valor o políticas. El proyecto implementa:

- Entornos de trading personalizados compatibles con la interfaz de Gymnasium
- Agentes basados en algoritmos como DQN (Deep Q-Network)
- Mecanismos de gestión de riesgo y optimización de cartera
- Herramientas de evaluación y visualización de resultados

## Descripción de la Fase 1: Infraestructura Base

Esta fase implementa la infraestructura básica del sistema de trading con aprendizaje por refuerzo. Se centra en establecer la estructura del proyecto, el sistema de configuración y las utilidades básicas necesarias para las fases posteriores.

## Estructura del Proyecto (Fase 1)

```
Fase1/
│
├── Agents/
│   └── .gitkeep           # Directorio para futuros agentes de RL
│
├── Config/
│   └── config.yaml        # Archivo de configuración con parámetros básicos
│
├── Data/
│   └── .gitkeep           # Directorio para datos financieros
│
├── Environments/
│   └── .gitkeep           # Directorio para entornos de trading
│
├── Utils/
│   └── helpers.py         # Funciones de utilidad básicas
│
├── main.py                # Script principal básico
│
├── requirements.txt       # Dependencias para la Fase 1
│
└── README.md              # Este archivo
```

## Componentes Implementados

### Sistema de Configuración (Config/config.yaml)

Archivo de configuración centralizado en formato YAML que contiene los parámetros básicos del sistema:

- **Configuración del entorno**: Parámetros como tamaño de ventana de observación, comisiones por operación y balance inicial.
- **Configuración de datos**: Especificación del símbolo a operar, fuente de datos, fechas de inicio y fin, y proporción de división entre datos de entrenamiento y prueba.
- **Configuración de logging**: Nivel de detalle, formato de los mensajes y directorio para guardar los registros.

### Script Principal (main.py)

Implementa la funcionalidad básica para demostrar la infraestructura del sistema:

- Carga de la configuración desde el archivo YAML
- Demostración de las funciones de normalización con datos aleatorios
- Cálculo de métricas financieras como el ratio de Sharpe
- Visualización de una curva de equity de ejemplo

### Utilidades Básicas (Utils/helpers.py)

Funciones de utilidad para el procesamiento de datos y visualización:

- **Funciones de normalización**: Implementa varios métodos de normalización (z-score, min-max, robust) para preparar los datos de entrada para los modelos de aprendizaje.
- **Funciones para métricas financieras**: Incluye el cálculo del ratio de Sharpe para evaluar el rendimiento ajustado al riesgo de las estrategias de trading.
- **Funciones de visualización**: Herramientas para graficar curvas de equity y otros indicadores de rendimiento.
- **Configuración de logging**: Sistema centralizado de registro para facilitar la depuración y el seguimiento del rendimiento.

## Requisitos

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- PyYAML

## Instalación

```bash
pip install -r requirements.txt
```

## Descripción de la Fase 2: Sistema de Datos

Esta fase implementa el sistema de carga y procesamiento de datos para el sistema de aprendizaje por refuerzo. Se centra en:

1. Cargar datos financieros de diferentes fuentes (Yahoo Finance, CSV)
2. Preprocesar los datos calculando indicadores técnicos
3. Dividir los datos en conjuntos de entrenamiento y prueba
4. Normalizar los datos para su uso posterior

## Estructura del Proyecto (Fase 2)

```
Fase2/
│
├── Agents/
│   └── .gitkeep           # Directorio para futuros agentes de RL (Fase 3)
│
├── Config/
│   └── config.yaml        # Archivo de configuración actualizado
│
├── Data/
│   └── data_loader.py     # Carga y preprocesamiento de datos financieros
│
├── Environments/
│   └── .gitkeep           # Directorio para entornos de trading (Fase 3)
│
├── Utils/
│   └── helpers.py         # Funciones de utilidad básicas
│
├── main.py                # Script principal con demostración de Fase 2
│
├── requirements.txt       # Dependencias actualizadas
│
└── README.md              # Este archivo
```

## Componentes Implementados en la Fase 2

### Cargador de Datos (Data/data_loader.py)

Clase para cargar y preprocesar datos financieros de diferentes fuentes:

- **Carga de datos**: Implementa métodos para cargar datos desde Yahoo Finance y archivos CSV.
- **Preprocesamiento**: Calcula indicadores técnicos como medias móviles, RSI, MACD y Bandas de Bollinger.
- **División de datos**: Divide los datos en conjuntos de entrenamiento y prueba según la configuración.
- **Normalización**: Proporciona métodos para normalizar los datos utilizando diferentes técnicas.

### Script Principal (main.py)

Implementa la funcionalidad para demostrar el sistema de datos:

- Carga de datos financieros desde diferentes fuentes
- Preprocesamiento de datos con cálculo de indicadores técnicos
- División de datos en conjuntos de entrenamiento y prueba
- Normalización de datos para su uso posterior
- Visualización básica de los datos cargados