"""
Renderizador de Poliedros 3D (Polyhedron Renderer).
Calcula la envolvente convexa (Convex Hull) a partir de una nube de puntos
(coordenadas espaciales de una capa de solvatación) y genera una representación visual
de alta resolución en 3D para su inclusión en publicaciones científicas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL Y PARÁMETROS GLOBALES
# ==============================================================================

FILE_INPUT = "vertices.xyz"
FILE_OUTPUT = "poliedro.png"

# Estética para artículo científico
FACE_COLOR = "#4F81BD"      # Azul corporativo/elegante
EDGE_COLOR = "black"
VERTEX_COLOR = "red"
ALPHA_LEVEL = 0.8           # Transparencia de las caras
DPI_QUALITY = 600           # Alta resolución para imprenta


# ==============================================================================
# 2. PROCESAMIENTO GEOMÉTRICO
# ==============================================================================

def load_vertices(filename: str) -> np.ndarray:
    """
    Lee las coordenadas espaciales ignorando símbolos químicos y cabeceras.
    Garantiza compatibilidad con archivos .xyz estándar.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[ERROR] Archivo de vértices '{filename}' no encontrado.")
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    coords = []
    # Detección automática de cabeceras .xyz
    start_idx = 2 if len(lines) > 0 and lines[0].strip().isdigit() else 0
        
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 3:
            # Asumimos que si hay 4 columnas, la primera es el símbolo (XYZ estándar)
            # Si hay 3, son directamente las coordenadas
            x_idx = 1 if len(parts) >= 4 else 0
            coords.append([float(parts[x_idx]), float(parts[x_idx+1]), float(parts[x_idx+2])])
            
    return np.array(coords)


def render_polyhedron(input_file: str, output_file: str):
    """
    Genera la envolvente convexa de la topología y renderiza el modelo 3D.
    """
    print(f"---> Generando malla poligonal a partir de: {input_file}")
    
    try:
        vertices = load_vertices(input_file)
    except Exception as e:
        print(e)
        return
        
    if len(vertices) < 4:
        print("[ERROR] Se necesitan al menos 4 vértices no coplanarios para formar un poliedro 3D.")
        return

    # Cálculo matemático de la envolvente convexa exterior
    hull = ConvexHull(vertices)

    # --------------------------------------------------------------------------
    # Renderizado 3D
    # --------------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extraer las caras del poliedro
    faces = [vertices[simplex] for simplex in hull.simplices]

    poly3d = Poly3DCollection(
        faces,
        facecolors=FACE_COLOR,
        edgecolors=EDGE_COLOR,
        linewidths=1.2,
        alpha=ALPHA_LEVEL
    )

    ax.add_collection3d(poly3d)

    # Dibujar los vértices (átomos de disolvente)
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        color=VERTEX_COLOR,
        s=30,
        zorder=5 # Asegura que los puntos se dibujen por encima de las caras
    )

    # Ajuste de la escala isométrica
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Estética de publicación (sin ejes, ángulo de cámara específico)
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)
    
    plt.tight_layout()

    # Exportación
    plt.savefig(
        output_file,
        dpi=DPI_QUALITY,
        bbox_inches="tight",
        transparent=True
    )
    plt.close()

    print(f" [OK] Envolvente convexa calculada ({len(hull.simplices)} caras triangulares).")
    print(f" [OK] Render 3D de alta resolución guardado como: '{output_file}'\n")


# ==============================================================================
# 3. EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    print("===================================================================")
    print(" Visualización 3D: Cálculo de Envolvente Convexa ")
    print("===================================================================\n")
    render_polyhedron(FILE_INPUT, FILE_OUTPUT)