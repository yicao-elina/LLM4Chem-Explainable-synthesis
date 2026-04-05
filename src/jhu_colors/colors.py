import json
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import os
import pkg_resources

def _load_colors():
    """Load JHU colors from the JSON file in the package data directory"""
    try:
        # Try to load from package data
        data_path = pkg_resources.resource_filename('jhu_colors', 'data/jhu_colors_rgb.json')
        with open(data_path, 'r') as f:
            return json.load(f)
    except:
        # Fallback: try to load from the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'data', 'jhu_colors_rgb.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            # If still not found, provide default colors
            return _get_default_colors()

def _get_default_colors():
    """Fallback default JHU colors if JSON file is not found"""
    return {
        "White": [1.0, 1.0, 1.0],
        "Double Black": [0.0, 0.0, 0.0],
        "Heritage Blue": [0.0, 0.1764705926179886, 0.4470588266849518],
        "Spirit Blue": [0.40784314274787903, 0.6745098233222961, 0.8980392217636108],
        "Red": [0.8117647171020508, 0.2705882489681244, 0.125490203499794],
        "Orange": [1.0, 0.6196078658103943, 0.10588235408067703],
        "Homewood Green": [0.0, 0.529411792755127, 0.40392157435417175],
        "Purple": [0.6431372761726379, 0.3607843220233917, 0.5960784554481506],
        "Gold": [0.9450980424880981, 0.7686274647712708, 0.0],
        "Forest Green": [0.15294118225574493, 0.3686274588108063, 0.239215686917305],
        "Harbor Blue": [0.3333333432674408, 0.5882353186607361, 0.8156862854957581],
        "Maroon": [0.4156862795352936, 0.125490203499794, 0.16862745583057404]
    }

def setup_jhu_colors():
    """Set up JHU colors as default matplotlib colors"""
    # Load colors
    jhu_colors = _load_colors()
    
    # Define color order
    color_order = [
        'Heritage Blue', 'Red', 'Homewood Green', 'Orange', 
        'Purple', 'Spirit Blue', 'Gold', 'Forest Green',
        'Harbor Blue', 'Maroon'
    ]
    
    # Create color list
    colors = [jhu_colors[name] for name in color_order if name in jhu_colors]
    
    # Set color cycle
    plt.rc('axes', prop_cycle=cycler(color=colors))
    
    # Optional: Set other defaults
    plt.rc('figure', figsize=(6, 4))
    plt.rc('axes', titlesize=16, labelsize=18)
    plt.rc('lines', linewidth=2)
    plt.rc('font', size=18)
    plt.rc('legend', fontsize=12)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # Line properties
    plt.rcParams['lines.linewidth'] = 1.5 # Slightly thicker lines for visibility

    # Axis and ticks
    plt.rcParams['axes.linewidth'] = 1 # Thicker axis lines
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.direction'] = 'in' # Inward ticks
    plt.rcParams['ytick.direction'] = 'in' # Inward ticks
    plt.rcParams['axes.grid'] = False # Disable grid by default
    # Save format (Nature often prefers PDF or EPS for vector graphics)
    plt.rcParams["savefig.format"] = 'png'
    plt.rcParams['savefig.dpi'] = 300 # High resolution for raster elements if any

    return jhu_colors

# Load colors and setup when module is imported
jhu_colors = setup_jhu_colors()

def get_jhu_color(name):
    """Get a specific JHU color by name"""
    return jhu_colors.get(name, [0, 0, 0])

def list_jhu_colors():
    """Print all available JHU colors"""
    print("Available JHU colors:")
    for name in sorted(jhu_colors.keys()):
        rgb = jhu_colors[name]
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        )
        print(f"  {name:<20} RGB: ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})  HEX: {hex_color}")

def show_jhu_palette():
    """Display all JHU colors in a plot"""
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_colors = sorted(jhu_colors.items())
    n_colors = len(sorted_colors)
    
    for i, (name, color) in enumerate(sorted_colors):
        y = n_colors - i - 1
        rect = plt.Rectangle((0, y), 4, 0.8, facecolor=color)
        ax.add_patch(rect)
        ax.text(4.2, y + 0.4, name, va='center', fontsize=12)
        
        rgb_text = f"RGB({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        ax.text(8, y + 0.4, rgb_text, va='center', fontsize=10, color='gray')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, n_colors)
    ax.set_title('JHU Color Palette', fontsize=18, pad=20)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Create colormaps
jhu_cmap = LinearSegmentedColormap.from_list(
    'jhu_gradient',
    [jhu_colors['Heritage Blue'], jhu_colors['White'], jhu_colors['Spirit Blue']]
)

jhu_diverging = LinearSegmentedColormap.from_list(
    'jhu_diverging',
    [jhu_colors['Red'], jhu_colors['White'], jhu_colors['Heritage Blue']]
)

jhu_sequential = LinearSegmentedColormap.from_list(
    'jhu_sequential',
    [jhu_colors['White'], jhu_colors['Spirit Blue'], jhu_colors['Heritage Blue']]
)

# Register colormaps
try:
    from matplotlib import colormaps
    colormaps.register(cmap=jhu_cmap, name='jhu')
    colormaps.register(cmap=jhu_diverging, name='jhu_diverging')
    colormaps.register(cmap=jhu_sequential, name='jhu_sequential')
except (ImportError, AttributeError):
    cm.register_cmap(name='jhu', cmap=jhu_cmap)
    cm.register_cmap(name='jhu_diverging', cmap=jhu_diverging)
    cm.register_cmap(name='jhu_sequential', cmap=jhu_sequential)