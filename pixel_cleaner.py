#!/usr/bin/env python3
"""
Grid Colorizer v5 - Converts any image into a pixel art mosaic.
- Accepts any image (with or without existing grid)
- Specify exact number of rows and columns
- Row and column numbers on margins
- Always square cells
- Reduced color palette

Usage: python grid_colorizer.py <image> --cols N --rows N [--colors N]
"""

import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from sklearn.cluster import KMeans


def get_cell_dominant_color(image_array, x1, y1, x2, y2, black_threshold=25):
    """
    Gets the dominant color of a region in the image.
    """
    cell = image_array[y1:y2, x1:x2]
    
    if cell.size == 0:
        return (0, 0, 0)
    
    # Flatten for analysis
    pixels = cell.reshape(-1, cell.shape[-1]) if len(cell.shape) == 3 else cell.reshape(-1, 1)
    
    # Calculate brightness
    if len(pixels[0]) >= 3:
        brightness = np.mean(pixels[:, :3], axis=1)
    else:
        brightness = pixels[:, 0]
    
    # Filter very dark pixels (possible grid borders from existing grid)
    non_black_mask = brightness > black_threshold
    non_black_pixels = pixels[non_black_mask]
    
    if len(non_black_pixels) == 0:
        return (0, 0, 0)
    
    # If most of the cell is black, return black
    if len(non_black_pixels) < len(pixels) * 0.1:
        return (0, 0, 0)
    
    # Calculate average color
    avg_color = np.mean(non_black_pixels[:, :3], axis=0).astype(int)
    return tuple(avg_color)


def create_color_palette(colors, n_colors=32):
    """
    Creates a reduced color palette using K-means clustering.
    """
    # Filter black/very dark colors
    non_black_colors = [c for c in colors if sum(c) > 75]
    
    if len(non_black_colors) < n_colors:
        n_colors = max(1, len(non_black_colors))
    
    if len(non_black_colors) == 0:
        return [(0, 0, 0)]
    
    colors_array = np.array(non_black_colors)
    
    # K-means to find representative colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(colors_array)
    
    palette = kmeans.cluster_centers_.astype(int)
    palette = np.vstack([[0, 0, 0], palette])
    
    return [tuple(c) for c in palette]


def find_closest_color(color, palette):
    """
    Finds the closest color in the palette using Euclidean distance.
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
    Returns white or black depending on background color for better readability.
    """
    brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
    return (255, 255, 255) if brightness < 128 else (0, 0, 0)


def process_image(input_path, output_path=None, n_cols=None, n_rows=None,
                  cell_size=None, border_width=2, n_colors=32, 
                  margin_size=30, show_numbers=True):
    """
    Processes any image and generates a gridded mosaic with reduced palette.
    
    Args:
        input_path: Input image path
        output_path: Output path (optional)
        n_cols: Number of columns (if specified, calculates cell size automatically)
        n_rows: Number of rows (if specified, calculates cell size automatically)
        cell_size: Size of each SQUARE cell in pixels (used if cols/rows not specified)
        border_width: Width of grid lines
        n_colors: Number of colors in palette
        margin_size: Margin size for numbers
        show_numbers: Whether to show row/column numbers
    """
    print(f"Loading image: {input_path}")
    img = Image.open(input_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    print(f"Original dimensions: {width}x{height}")
    
    # Calculate cell size based on desired columns/rows or use provided cell_size
    if n_cols is not None and n_rows is not None:
        # User specified exact grid dimensions
        # Calculate cell size to fit the image, using the smaller dimension
        # to ensure square cells
        cell_w = width / n_cols
        cell_h = height / n_rows
        cell_size = int(min(cell_w, cell_h))
        
        # Recalculate to maintain aspect ratio with square cells
        # Use the specified dimensions directly
        print(f"Target grid: {n_cols} columns x {n_rows} rows")
    elif n_cols is not None:
        # Only columns specified, calculate rows to maintain aspect ratio
        cell_size = width // n_cols
        n_rows = height // cell_size
        print(f"Columns specified: {n_cols}, calculated rows: {n_rows}")
    elif n_rows is not None:
        # Only rows specified, calculate columns to maintain aspect ratio
        cell_size = height // n_rows
        n_cols = width // cell_size
        print(f"Rows specified: {n_rows}, calculated columns: {n_cols}")
    else:
        # Use cell_size to calculate grid
        if cell_size is None:
            cell_size = 20
        n_cols = width // cell_size
        n_rows = height // cell_size
    
    # Ensure minimum cell size
    cell_size = max(cell_size, 5)
    
    print(f"Cell size: {cell_size}x{cell_size} px (square)")
    print(f"Grid: {n_cols} columns x {n_rows} rows")
    print(f"Total cells: {n_cols * n_rows}")
    
    # First pass: extract colors from each cell
    print("Extracting colors from each cell...")
    cell_colors = {}
    all_colors = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate coordinates in the original image
            # Distribute evenly across the image
            x1 = int(col * width / n_cols)
            y1 = int(row * height / n_rows)
            x2 = int((col + 1) * width / n_cols)
            y2 = int((row + 1) * height / n_rows)
            
            dominant_color = get_cell_dominant_color(img_array, x1, y1, x2, y2)
            cell_colors[(row, col)] = dominant_color
            
            if sum(dominant_color) > 75:
                all_colors.append(dominant_color)
    
    print(f"Unique colors extracted: {len(set(all_colors))}")
    
    # Create reduced palette
    print(f"Creating palette with {n_colors} colors...")
    palette = create_color_palette(all_colors, n_colors)
    print(f"Palette created with {len(palette)} colors")
    
    # Map colors to palette
    color_mapping = {}
    for original_color in set(cell_colors.values()):
        color_mapping[original_color] = find_closest_color(original_color, palette)
    
    # Calculate output image dimensions
    # Each cell has: cell_size + border_width (for right/bottom border)
    grid_width = n_cols * cell_size + (n_cols + 1) * border_width
    grid_height = n_rows * cell_size + (n_rows + 1) * border_width
    
    if show_numbers:
        total_width = grid_width + margin_size
        total_height = grid_height + margin_size
        offset_x = margin_size
        offset_y = margin_size
    else:
        total_width = grid_width
        total_height = grid_height
        offset_x = 0
        offset_y = 0
    
    print(f"Output dimensions: {total_width}x{total_height}")
    
    # Create output image with black background
    output_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))
    draw = ImageDraw.Draw(output_img)
    
    # Try to load a font, use default if it fails
    try:
        font_size = min(margin_size - 4, cell_size - 2, 14)
        font_size = max(font_size, 8)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw cells
    print("Drawing cells...")
    for row in range(n_rows):
        for col in range(n_cols):
            original_color = cell_colors[(row, col)]
            quantized_color = color_mapping[original_color]
            
            # Coordinates in output image
            x1 = offset_x + border_width + col * (cell_size + border_width)
            y1 = offset_y + border_width + row * (cell_size + border_width)
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Draw cell
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=quantized_color)
    
    # Draw column numbers (top)
    if show_numbers:
        print("Drawing column numbers...")
        # Calculate step for showing numbers (show all if grid is small)
        col_step = 1
        if n_cols > 30:
            col_step = 5
        if n_cols > 60:
            col_step = 10
        
        for col in range(n_cols):
            # Always show first, last, and every col_step
            if col_step > 1 and col != 0 and col != n_cols - 1 and (col + 1) % col_step != 0:
                continue
                
            x = offset_x + border_width + col * (cell_size + border_width) + cell_size // 2
            y = margin_size // 2
            
            text = str(col + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text((x - text_width // 2, y - text_height // 2), text, 
                     fill=(200, 200, 200), font=font)
    
    # Draw row numbers (left)
    if show_numbers:
        print("Drawing row numbers...")
        row_step = 1
        if n_rows > 30:
            row_step = 5
        if n_rows > 60:
            row_step = 10
            
        for row in range(n_rows):
            # Always show first, last, and every row_step
            if row_step > 1 and row != 0 and row != n_rows - 1 and (row + 1) % row_step != 0:
                continue
                
            x = margin_size // 2
            y = offset_y + border_width + row * (cell_size + border_width) + cell_size // 2
            
            text = str(row + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text((x - text_width // 2, y - text_height // 2), text, 
                     fill=(200, 200, 200), font=font)
    
    # Save image
    if output_path is None:
        if '.' in input_path:
            base, ext = input_path.rsplit('.', 1)
            output_path = f"{base}_grid_{n_cols}x{n_rows}_{n_colors}colors.{ext}"
        else:
            output_path = f"{input_path}_grid_{n_cols}x{n_rows}_{n_colors}colors.png"
    
    output_img.save(output_path, quality=95)
    print(f"\nImage saved: {output_path}")
    
    # Statistics
    unique_colors_used = len(set(color_mapping.values()))
    print(f"Unique colors in result: {unique_colors_used}")
    print(f"Total cells: {n_rows * n_cols}")
    
    return output_path, palette


def main():
    parser = argparse.ArgumentParser(
        description='Converts any image into a pixel art mosaic with reduced color palette.'
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', nargs='?', help='Output image path (optional)')
    parser.add_argument('--cols', type=int, 
                       help='Number of columns in the grid')
    parser.add_argument('--rows', type=int, 
                       help='Number of rows in the grid')
    parser.add_argument('--colors', type=int, default=32, 
                       help='Number of colors in palette (default: 32)')
    parser.add_argument('--cell-size', type=int, default=20, 
                       help='Cell size in pixels, used if cols/rows not specified (default: 20)')
    parser.add_argument('--border', type=int, default=2, 
                       help='Border width between cells (default: 2)')
    parser.add_argument('--margin', type=int, default=30, 
                       help='Margin size for numbers (default: 30)')
    parser.add_argument('--no-numbers', action='store_true', 
                       help='Hide row/column numbers')
    
    args = parser.parse_args()
    
    try:
        output_path, palette = process_image(
            args.input, 
            args.output, 
            n_cols=args.cols,
            n_rows=args.rows,
            cell_size=args.cell_size,
            border_width=args.border,
            n_colors=args.colors,
            margin_size=args.margin,
            show_numbers=not args.no_numbers
        )
        print(f"\nProcess completed successfully!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
