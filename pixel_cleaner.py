#!/usr/bin/env python3
"""
Grid Colorizer v4 - Convierte cualquier imagen en un mosaico cuadriculado.
- Acepta cualquier imagen (con o sin cuadrícula previa)
- Enumera filas y columnas en los bordes
- Celdas siempre cuadradas
- Paleta de colores reducida

Uso: python grid_colorizer.py <imagen> [--colors N] [--cell-size N]
"""

import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from sklearn.cluster import KMeans


def get_cell_dominant_color(image_array, x1, y1, x2, y2, black_threshold=25):
    """
    Obtiene el color dominante de una región de la imagen.
    """
    cell = image_array[y1:y2, x1:x2]
    
    if cell.size == 0:
        return (0, 0, 0)
    
    # Aplanar para análisis
    pixels = cell.reshape(-1, cell.shape[-1]) if len(cell.shape) == 3 else cell.reshape(-1, 1)
    
    # Calcular brillo
    if len(pixels[0]) >= 3:
        brightness = np.mean(pixels[:, :3], axis=1)
    else:
        brightness = pixels[:, 0]
    
    # Filtrar píxeles muy oscuros (posibles bordes de cuadrícula existente)
    non_black_mask = brightness > black_threshold
    non_black_pixels = pixels[non_black_mask]
    
    if len(non_black_pixels) == 0:
        return (0, 0, 0)
    
    # Si la mayoría de la celda es negra, devolver negro
    if len(non_black_pixels) < len(pixels) * 0.1:
        return (0, 0, 0)
    
    # Calcular color promedio
    avg_color = np.mean(non_black_pixels[:, :3], axis=0).astype(int)
    return tuple(avg_color)


def create_color_palette(colors, n_colors=32):
    """
    Crea una paleta reducida de colores usando K-means clustering.
    """
    # Filtrar colores negros/muy oscuros
    non_black_colors = [c for c in colors if sum(c) > 75]
    
    if len(non_black_colors) < n_colors:
        n_colors = max(1, len(non_black_colors))
    
    if len(non_black_colors) == 0:
        return [(0, 0, 0)]
    
    colors_array = np.array(non_black_colors)
    
    # K-means para encontrar colores representativos
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(colors_array)
    
    palette = kmeans.cluster_centers_.astype(int)
    palette = np.vstack([[0, 0, 0], palette])
    
    return [tuple(c) for c in palette]


def find_closest_color(color, palette):
    """
    Encuentra el color más cercano en la paleta.
    """
    if color == (0, 0, 0) or sum(color) < 75:
        return (0, 0, 0)
    
    color_array = np.array(color)
    min_dist = float('inf')
    closest = palette[0]
    
    for p_color in palette:
        if p_color == (0, 0, 0):
            continue
        dist = np.sqrt(np.sum((color_array - np.array(p_color)) ** 2))
        if dist < min_dist:
            min_dist = dist
            closest = p_color
    
    return closest


def get_text_color(bg_color):
    """
    Devuelve blanco o negro según el color de fondo para mejor legibilidad.
    """
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    return (255, 255, 255) if brightness < 128 else (0, 0, 0)


def process_image(input_path, output_path=None, cell_size=20, border_width=2, 
                  n_colors=32, margin_size=30, show_numbers=True):
    """
    Procesa cualquier imagen y genera un mosaico cuadriculado con paleta reducida.
    
    Args:
        input_path: Ruta de la imagen de entrada
        output_path: Ruta de salida (opcional)
        cell_size: Tamaño de cada celda CUADRADA en píxeles
        border_width: Ancho de las líneas de la cuadrícula
        n_colors: Número de colores en la paleta
        margin_size: Tamaño del margen para los números
        show_numbers: Si mostrar números de fila/columna
    """
    print(f"Cargando imagen: {input_path}")
    img = Image.open(input_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    print(f"Dimensiones originales: {width}x{height}")
    
    # Calcular número de celdas (siempre cuadradas)
    # El tamaño efectivo de celda incluye el borde
    effective_cell_size = cell_size + border_width
    
    n_cols = width // cell_size
    n_rows = height // cell_size
    
    print(f"Tamaño de celda: {cell_size}x{cell_size} px (cuadradas)")
    print(f"Cuadrícula: {n_cols} columnas x {n_rows} filas")
    
    # Primera pasada: extraer colores de cada celda
    print("Extrayendo colores de cada celda...")
    cell_colors = {}
    all_colors = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Coordenadas en la imagen original
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = min(x1 + cell_size, width)
            y2 = min(y1 + cell_size, height)
            
            dominant_color = get_cell_dominant_color(img_array, x1, y1, x2, y2)
            cell_colors[(row, col)] = dominant_color
            
            if sum(dominant_color) > 75:
                all_colors.append(dominant_color)
    
    print(f"Colores únicos extraídos: {len(set(all_colors))}")
    
    # Crear paleta reducida
    print(f"Creando paleta de {n_colors} colores...")
    palette = create_color_palette(all_colors, n_colors)
    print(f"Paleta creada con {len(palette)} colores")
    
    # Mapear colores a la paleta
    color_mapping = {}
    for original_color in set(cell_colors.values()):
        color_mapping[original_color] = find_closest_color(original_color, palette)
    
    # Calcular dimensiones de la imagen de salida
    # Cada celda tiene: cell_size + border_width (para el borde derecho/inferior)
    grid_width = n_cols * cell_size + (n_cols + 1) * border_width
    grid_height = n_rows * cell_size + (n_rows + 1) * border_width
    
    if show_numbers:
        total_width = grid_width + margin_size  # Margen izquierdo para números de fila
        total_height = grid_height + margin_size  # Margen superior para números de columna
        offset_x = margin_size
        offset_y = margin_size
    else:
        total_width = grid_width
        total_height = grid_height
        offset_x = 0
        offset_y = 0
    
    print(f"Dimensiones de salida: {total_width}x{total_height}")
    
    # Crear imagen de salida con fondo negro
    output_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))
    draw = ImageDraw.Draw(output_img)
    
    # Intentar cargar una fuente, usar default si falla
    try:
        font_size = min(margin_size - 4, cell_size - 2, 14)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Dibujar las celdas
    print("Dibujando celdas...")
    for row in range(n_rows):
        for col in range(n_cols):
            original_color = cell_colors[(row, col)]
            quantized_color = color_mapping[original_color]
            
            # Coordenadas en la imagen de salida
            # Cada celda empieza después del borde izquierdo/superior
            x1 = offset_x + border_width + col * (cell_size + border_width)
            y1 = offset_y + border_width + row * (cell_size + border_width)
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Dibujar celda
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=quantized_color)
    
    # Dibujar números de columna (arriba)
    # Mostrar solo cada N columnas si hay muchas
    if show_numbers:
        print("Dibujando números de columna...")
        # Calcular cada cuántas columnas mostrar número
        col_step = 1
        if n_cols > 50:
            col_step = 5
        elif n_cols > 100:
            col_step = 10
        
        for col in range(n_cols):
            if col_step > 1 and (col + 1) % col_step != 0 and col != 0:
                continue
                
            x = offset_x + border_width + col * (cell_size + border_width) + cell_size // 2
            y = margin_size // 2
            
            text = str(col + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text((x - text_width // 2, y - text_height // 2), text, 
                     fill=(200, 200, 200), font=font)
    
    # Dibujar números de fila (izquierda)
    # Mostrar solo cada N filas si hay muchas
    if show_numbers:
        print("Dibujando números de fila...")
        row_step = 1
        if n_rows > 50:
            row_step = 5
        elif n_rows > 100:
            row_step = 10
            
        for row in range(n_rows):
            if row_step > 1 and (row + 1) % row_step != 0 and row != 0:
                continue
                
            x = margin_size // 2
            y = offset_y + border_width + row * (cell_size + border_width) + cell_size // 2
            
            text = str(row + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text((x - text_width // 2, y - text_height // 2), text, 
                     fill=(200, 200, 200), font=font)
    
    # Guardar imagen
    if output_path is None:
        if '.' in input_path:
            base, ext = input_path.rsplit('.', 1)
            output_path = f"{base}_grid_{n_colors}colors.{ext}"
        else:
            output_path = f"{input_path}_grid_{n_colors}colors.png"
    
    output_img.save(output_path, quality=95)
    print(f"\nImagen guardada: {output_path}")
    
    # Estadísticas
    unique_colors_used = len(set(color_mapping.values()))
    print(f"Colores únicos en resultado: {unique_colors_used}")
    print(f"Celdas totales: {n_rows * n_cols}")
    
    return output_path, palette


def main():
    parser = argparse.ArgumentParser(
        description='Convierte cualquier imagen en un mosaico cuadriculado con paleta reducida.'
    )
    parser.add_argument('input', help='Ruta de la imagen de entrada')
    parser.add_argument('output', nargs='?', help='Ruta de la imagen de salida (opcional)')
    parser.add_argument('--colors', type=int, default=32, 
                       help='Número de colores en la paleta (default: 32)')
    parser.add_argument('--cell-size', type=int, default=20, 
                       help='Tamaño de cada celda cuadrada en píxeles (default: 20)')
    parser.add_argument('--border', type=int, default=2, 
                       help='Ancho del borde entre celdas (default: 2)')
    parser.add_argument('--margin', type=int, default=30, 
                       help='Tamaño del margen para números (default: 30)')
    parser.add_argument('--no-numbers', action='store_true', 
                       help='No mostrar números de fila/columna')
    
    args = parser.parse_args()
    
    try:
        output_path, palette = process_image(
            args.input, 
            args.output, 
            cell_size=args.cell_size,
            border_width=args.border,
            n_colors=args.colors,
            margin_size=args.margin,
            show_numbers=not args.no_numbers
        )
        print(f"\n¡Proceso completado exitosamente!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
