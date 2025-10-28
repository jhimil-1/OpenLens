import requests
import json
from PIL import Image
import io

# Create a test image
test_image = Image.new('RGB', (200, 200), color='blue')
img_byte_arr = io.BytesIO()
test_image.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

# Upload to the search endpoint
files = {'image': ('test_image.png', img_byte_arr.getvalue(), 'image/png')}
response = requests.post('http://localhost:8000/search', files=files)

print('Status Code:', response.status_code)
if response.status_code == 200:
    result = response.json()
    print(f'Found {len(result["results"])} similar products')
    print(f'Source: {result["source"]}')
    if result['results']:
        print('First result:')
        print(json.dumps(result['results'][0], indent=2))
else:
    print('Error:', response.text)