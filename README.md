# Simulaciones Computacionales en Agregados Moleculares de Rb+(H2)N Inmersos en Medios Cuánticos

**Autor:** Ainhoa Barjola Ruiz  
**Institución:** Universidad Alfonso X el Sabio (UAX) / Instituto de Física Fundamental (IFF-CSIC)  
**Tutorización:** José Antonio Prieto Persiguero, Tomás González Lezana y Mª Judit Montes De Oca Estévez

Este repositorio contiene el ecosistema algorítmico y las herramientas de análisis de datos desarrolladas para el Trabajo de Fin de Grado en Física: *"Simulaciones Computacionales en Agregados Moleculares de Rb+(H2)N inmersos en Medios Cuánticos"*. 

El código está desarrollado íntegramente en Python y aborda la exploración del espacio de fases, la optimización topológica y el análisis termodinámico de nanogotas cuánticas mediante la aproximación del pseudo-átomo y el potencial analítico Improved Lennard-Jones (ILJ).

## 📂 Estructura del Repositorio

El código fuente está dividido en tres módulos principales orientados a la física del problema:

### `/src` (Núcleo de Optimización y Termodinámica)
Contiene los motores de búsqueda global y minimización de energía:
* `evolutionary_algorithm.py`: Motor genético continuo para ensamblaje de clústeres.
* `constrained_evolution.py`: Optimizador con restricción espacial (capas fijas) para tamaños macroscópicos.
* `global_relaxation.py`: Minimización sin restricciones para auditar la estabilidad de macroestructuras.
* `inverse_optimization.py`: Secuencia iterativa de decapado para trazar curvas energéticas.
* `lambda_energy_scan.py`: Exploración unidimensional de la SEP mediante homotecias.

### `/analysis` (Análisis Físico y Termodinámico)
Scripts dedicados al procesamiento de datos crudos y visualización científica:
* `plot_evaporation_energy.py`: Generador de la curva de evaporación para la identificación de números mágicos.
* `radial_energy_scan.py`: Cálculo de gaussianas y funciones de distribución radial.
* `gaussian_broadening.py`: Cálculo de la Función de Distribución Radial mediante ensanchamiento gaussiano de las distancias atómicas para la identificación de capas de solvatación.

### `/utils` (Herramientas de Topología y Geometría)
Utilidades para la preparación y renderizado de coordenadas moleculares:
* `radial_scaling_tool.py`: Herramienta para dividir sistemas en capas y aplicar escalas interactivas.
* `spherical_projection.py`: Implementación de máscaras de selección para colapsos radiales esféricos.
* `translate_to_origin.py`: Estandarización de coordenadas respecto al centro de fuerzas.
* `layer_colorizer.py`: Algoritmo K-Means para la segregación de capas y exportación a formato `.cml` para renderizado fotorrealista en Avogadro.
* `polyhedron_renderer.py`: Cálculo de la envolvente convexa (Convex Hull) y renderizado fotorrealista de poliedros en 3D.

## ⚙️ Requisitos (Dependencias)
Para ejecutar los scripts de este repositorio se requiere un entorno de Python 3 con las siguientes librerías:
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
