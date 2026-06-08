"""
Análisis Termodinámico y Visualización (Evaporation Energy Plotter).
Genera la curva de energía de evaporación frente al tamaño del clúster (N)
para identificar los 'números mágicos' (picos de estabilidad) asociados al 
cierre de las capas de solvatación del sistema Rb+(H2)_N.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# ==============================================================================
# 1. PARÁMETROS GLOBALES Y CONFIGURACIÓN VISUAL
# ==============================================================================
COLUMN_N = 'N'
COLUMN_E_EVAP = 'Energía de evaporación'

# Estética de la gráfica (estilo artículo científico)
PLOT_COLOR = 'teal'
MARKER_STYLE = 'o'
LINE_STYLE = '-'


# ==============================================================================
# 2. PROCESAMIENTO Y VISUALIZACIÓN DE DATOS
# ==============================================================================

def plot_evaporation_curve(file_path: str, sheet_name: str, n_min: Optional[int] = None, n_max: Optional[int] = None) -> None:
    """
    Lee los datos termodinámicos tabulados, filtra el rango espacial de N
    y renderiza la función de energía de evaporación.
    """
    try:
        # Cargar los datos, ignorando la primera fila vacía (header=1)
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        
        # Limpieza de las cabeceras para evitar errores por espacios en blanco
        df.columns = df.columns.astype(str).str.strip()

        if COLUMN_N not in df.columns or COLUMN_E_EVAP not in df.columns:
            print(f"\n[ERROR] Columnas '{COLUMN_N}' o '{COLUMN_E_EVAP}' no encontradas en la hoja '{sheet_name}'.")
            print(f"Columnas detectadas: {list(df.columns)}")
            return

        # Limpiar valores nulos
        df_clean = df.dropna(subset=[COLUMN_N, COLUMN_E_EVAP])
        
        n_min_total = int(df_clean[COLUMN_N].min())
        n_max_total = int(df_clean[COLUMN_N].max())

        # Asignar límites por defecto si no se especifican
        n_min_plot = n_min if n_min is not None else n_min_total
        n_max_plot = n_max if n_max is not None else n_max_total

        # Filtrar el dominio de N a representar
        df_filtered = df_clean[(df_clean[COLUMN_N] >= n_min_plot) & (df_clean[COLUMN_N] <= n_max_plot)]

        if df_filtered.empty:
            print(f"\n[AVISO] No hay datos en el rango N = [{n_min_plot}, {n_max_plot}].")
            return

        # ----------------------------------------------------------------------
        # Renderizado de la Gráfica
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        
        plt.plot(df_filtered[COLUMN_N], df_filtered[COLUMN_E_EVAP], 
                 marker=MARKER_STYLE, linestyle=LINE_STYLE, color=PLOT_COLOR, markersize=6)

        # ¡Corregido el N2 a H2!
        plt.title(f'Energía de Evaporación del Agregado $Rb^+(H_2)_N$\n(Topología: {sheet_name} | Rango: $N={n_min_plot}$ a ${n_max_plot}$)', 
                  fontsize=14, fontweight='bold', pad=15)
                  
        plt.xlabel('Número de moléculas de disolvente ($N$)', fontsize=12)
        plt.ylabel('Energía de evaporación ($E_{evap}$ / meV)', fontsize=12)
        
        # Forzar que el eje X muestre todos los enteros para identificar fácilmente los números mágicos
        plt.xticks(range(n_min_plot, n_max_plot + 1))
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n[ERROR] Ha ocurrido un fallo en la generación de la gráfica: {e}")


# ==============================================================================
# 3. INTERFAZ DE USUARIO (CLI)
# ==============================================================================

def interactive_plotter():
    print("===================================================================")
    print(" Herramienta de Visualización: Energías de Evaporación ")
    print("===================================================================\n")
    
    file_path = input("-> Introduce la ruta completa del archivo Excel: ").strip()
    
    try:
        xls = pd.ExcelFile(file_path)
        print("\n[INFO] Topologías (pestañas) disponibles en el registro:")
        for sheet in xls.sheet_names:
            print(f"  - {sheet}")

        sheet_name = input("\n-> Escribe el nombre exacto de la pestaña a procesar: ").strip()
        
        # Validar la existencia de la hoja
        if sheet_name not in xls.sheet_names:
            print(f"[ERROR] La pestaña '{sheet_name}' no existe en el documento.")
            return

        # Cargar momentáneamente para ver el rango total
        df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        df_temp.columns = df_temp.columns.astype(str).str.strip()
        
        if COLUMN_N in df_temp.columns:
            df_clean = df_temp.dropna(subset=[COLUMN_N])
            print(f"\n[INFO] Rango termodinámico disponible: N = {int(df_clean[COLUMN_N].min())} hasta {int(df_clean[COLUMN_N].max())}.")
        
        val_min = input("-> Límite inferior de N (Enter para usar el mínimo): ").strip()
        val_max = input("-> Límite superior de N (Enter para usar el máximo): ").strip()
        
        n_min = int(val_min) if val_min else None
        n_max = int(val_max) if val_max else None
        
        print("\n[INFO] Procesando datos y renderizando gráfica...")
        plot_evaporation_curve(file_path, sheet_name, n_min, n_max)

    except FileNotFoundError:
        print(f"\n[ERROR] Archivo no localizado. Verifica la ruta: '{file_path}'")
    except ValueError:
        print("\n[ERROR] Debes introducir valores enteros válidos para el rango de N.")


if __name__ == "__main__":
    interactive_plotter()