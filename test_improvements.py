import requests
import json
from PIL import Image
import io

# Create a simple black rectangle with light gray screen (electronics-like)
img = Image.new('RGB', (224, 224), color='black')
# Add a light gray "screen" area
for y in range(50, 174):
    for x in range(30, 194):
        img.putpixel((x, y), (200, 200, 200))

# Save to bytes
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Test the search endpoint
files = {'image': ('test_electronics.jpg', img_bytes.getvalue(), 'image/jpeg')}
data = {'query': 'headphones electronics'}

response = requests.post('http://localhost:8002/search', files=files, data=data)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Detected Category: {result.get('detected_category', 'N/A')}")
    print(f"Detected Attributes: {result.get('detected_attributes', {})}")
    print(f"Total Products Found: {len(result.get('results', []))}")
    
    if result.get('results'):
        print("\nTop Results:")
        for i, product in enumerate(result['results'][:3]):
            print(f"{i+1}. {product['title'][:60]}...")
            print(f"   Similarity: {product['similarity']:.3f}")
            print(f"   Source: {product['source']}")
            print()
else:
    print(f"Error: {response.text}")