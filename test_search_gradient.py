import requests
import json
from PIL import Image
import io

# Create a more realistic test image (gradient pattern that might match products)
width, height = 300, 300
test_image = Image.new('RGB', (width, height))

# Create a gradient pattern
for x in range(width):
    for y in range(height):
        r = int(255 * x / width)
        g = int(255 * y / height)
        b = int(255 * (x + y) / (width + height))
        test_image.putpixel((x, y), (r, g, b))

img_byte_arr = io.BytesIO()
test_image.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

# Upload to the search endpoint
files = {'image': ('gradient_product.png', img_byte_arr.getvalue(), 'image/png')}
response = requests.post('http://localhost:8000/search', files=files)

print('Status Code:', response.status_code)
if response.status_code == 200:
    result = response.json()
    print(f'Found {len(result["results"])} similar products')
    print(f'Source: {result["source"]}')
    print('\nTop 5 results:')
    for i, product in enumerate(result['results'][:5], 1):
        print(f"{i}. {product['title']} - {product['price']} - Similarity: {product['similarity']:.3f}")
        print(f"   Image: {product['image_url'][:80]}...")
        print()
else:
    print('Error:', response.text)