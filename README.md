# Trading Adaptativo con Aprendizaje por Refuerzo - Fase 1

## Descripción de la Fase 1: Infraestructura Base

Esta fase implementa la infraestructura básica del sistema de trading con aprendizaje por refuerzo. Se centra en establecer la estructura del proyecto, el sistema de configuración y las utilidades básicas necesarias para las fases posteriores.

## Estructura del Proyecto (Fase 1)

```
Fase1/
│
├── Agents/
│   └── .gitkeep            # Carpeta para futuros agentes de RL
│
├── Config/
│   └── config.yaml        # Archivo de configuración con parámetros básicos
│
├── Data/
│   └── .gitkeep            # Carpeta para futuros datos financieros
│
├── Environments/
│   └── .gitkeep            # Carpeta para futuros entornos de trading
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

- Configuración del entorno (parámetros básicos)
- Configuración de datos (fuente, fechas, etc.)
- Configuración de logging

### Utilidades Básicas (Utils/helpers.py)

Funciones de utilidad para el procesamiento de datos y visualización:

- Funciones de normalización
- Funciones para métricas financieras básicas
- Configuración de logging

### Carpetas Preparadas para Fases Futuras

- **Agents/**: Contendrá los agentes de aprendizaje por refuerzo (DQN, etc.)
- **Data/**: Contendrá los módulos de carga y procesamiento de datos financieros
- **Environments/**: Contendrá los entornos de trading personalizados basados en Gymnasium

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

## Próximas Fases

- **Fase 2**: Implementación del entorno de trading y carga de datos
- **Fase 3**: Implementación del agente DQN y entrenamiento
- **Fase 4**: Evaluación, visualización y optimización