import os
import pytest
from PIL import Image
import io
import numpy as np

# Ensure main doesn't try to load heavy models during import
os.environ['SKIP_MODEL_INIT'] = '1'

from main import extract_dominant_colors, get_color_name


def make_solid_image(color, size=(128, 128)):
    """Create a solid RGB PIL image from an (R,G,B) tuple or name."""
    if isinstance(color, str):
        # simple named colors
        color_map = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (220, 30, 30),
            'dark_red': (110, 20, 20),
        }
        rgb = color_map.get(color, (128, 128, 128))
    else:
        rgb = tuple(color)
    return Image.new('RGB', size, rgb)


def test_detect_black():
    img = make_solid_image('black')
    colors = extract_dominant_colors(img)
    assert isinstance(colors, list)
    assert 'black' in colors or colors == ['black']


def test_detect_white():
    img = make_solid_image('white')
    colors = extract_dominant_colors(img)
    assert isinstance(colors, list)
    assert 'white' in colors or colors == ['white']


def test_detect_red():
    img = make_solid_image('red')
    colors = extract_dominant_colors(img)
    assert isinstance(colors, list)
    assert 'red' in colors


def test_dark_red_not_black():
    img = make_solid_image('dark_red')
    colors = extract_dominant_colors(img)
    # dark red should be red (or multi-color) but not misclassified as black
    assert 'black' not in colors
    assert 'red' in colors or colors == ['gray']


if __name__ == '__main__':
    pytest.main([__file__])
