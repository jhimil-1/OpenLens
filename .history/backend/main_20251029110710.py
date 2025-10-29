from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import open_clip
from PIL import Image
import io
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
import uuid
import numpy as np
import base64
import asyncio
import time
from datetime import datetime, timedelta
from collections import defaultdict
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

load_dotenv()

# Rate limiting and caching system
class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 100):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)  # Store timestamps by API key
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, api_key: str):
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old requests
            self.requests[api_key] = [req_time for req_time in self.requests[api_key] if req_time > minute_ago]
            
            # Check if we need to wait
            if len(self.requests[api_key]) >= self.max_requests:
                # Calculate wait time until oldest request is outside the window
                oldest_request = min(self.requests[api_key])
                wait_time = (oldest_request + 60) - now
                if wait_time > 0:
                    print(f"[RATE LIMIT] Waiting {wait_time:.1f}s to avoid API limits")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.requests[api_key].append(now)

class SearchCache:
    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = asyncio.Lock()
    
    def _generate_key(self, query: str, page: int) -> str:
        return f"{query.lower().strip()}_{page}"
    
    async def get(self, query: str, page: int) -> Optional[List[dict]]:
        async with self.lock:
            key = self._generate_key(query, page)
            if key in self.cache:
                timestamp, data = self.cache[key]
                if datetime.now() - timestamp < self.ttl:
                    print(f"[CACHE] Hit for '{query}' page {page}")
                    return data
                else:
                    del self.cache[key]
            return None
    
    async def set(self, query: str, page: int, data: List[dict]):
        async with self.lock:
            key = self._generate_key(query, page)
            self.cache[key] = (datetime.now(), data)
            print(f"[CACHE] Stored '{query}' page {page}")

# Initialize rate limiter and cache
rate_limiter = RateLimiter(max_requests_per_minute=90)  # Conservative limit
search_cache = SearchCache(ttl_hours=24)

class APIKeyManager:
    def __init__(self):
        self.keys = []
        self.current_index = 0
        self.load_keys()
    
    def load_keys(self):
        """Load all available API keys from environment"""
        # Primary key
        primary_key = os.getenv("GOOGLE_API_KEY")
        if primary_key:
            self.keys.append(primary_key)
        
        # Backup keys (can add more as needed)
        backup_keys = [
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3"),
            "AIzaSyBn8XLzv_GM18y8MLoVUkHT-F9yQIoeaH0",  # Hardcoded backup
            "AIzaSyAmkVShutgc_MBRFSH43WY7o8SMqlhntsc"   # Another backup
        ]
        
        for key in backup_keys:
            if key and key not in self.keys:
                self.keys.append(key)
        
        print(f"[API KEYS] Loaded {len(self.keys)} API keys")
    
    def get_current_key(self) -> str:
        """Get current API key"""
        if not self.keys:
            return None
        return self.keys[self.current_index]
    
    def rotate_key(self) -> str:
        """Rotate to next API key"""
        if not self.keys:
            return None
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"[API KEYS] Rotated to key {self.current_index + 1}/{len(self.keys)}")
        return self.keys[self.current_index]
    
    def get_all_keys(self) -> List[str]:
        """Get all available keys"""
        return self.keys.copy()

# Initialize API key manager
api_key_manager = APIKeyManager()

# Initialize MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.visual_search
collection_collection = db.image_collections  # Collection for storing image collections

app = FastAPI(title="Visual Product Search API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device)

# Initialize Qdrant Cloud client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "product_embeddings"

# Pydantic models
class ProductResult(BaseModel):
    title: str
    image_url: str
    price: Optional[str] = None
    description: Optional[str] = None
    link: str
    similarity: float
    source: str

class CollectionItem(BaseModel):
    image_url: str
    collection_id: str
    created_at: datetime

class CollectionResponse(BaseModel):
    items: List[CollectionItem]
    total: int

class SearchResponse(BaseModel):
    results: List[ProductResult]
    total_scraped: int
    sources: List[str]
    detected_category: Optional[str] = None
    detected_attributes: Optional[dict] = None

# Initialize collection on startup
@app.on_event("startup")
async def startup_event():
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        if not any(col.name == COLLECTION_NAME for col in collections):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"[OK] Created collection: {COLLECTION_NAME}")
        else:
            print(f"[OK] Collection {COLLECTION_NAME} already exists")
    except Exception as e:
        print(f"[ERROR] Error initializing collection: {e}")

def extract_dominant_colors(image: Image.Image, num_colors: int = 3) -> List[str]:
    """Extract dominant colors from image with aggressive red detection"""
    try:
        image = image.convert("RGB")
        image = image.resize((100, 100))  # Smaller for faster processing
        
        pixels = np.array(image).reshape(-1, 3)
        
        # Sample pixels for faster processing
        if len(pixels) > 3000:
            indices = np.random.choice(len(pixels), 3000, replace=False)
            pixels = pixels[indices]
        
        # Aggressive red detection - check if red is dominant
        red_pixels = 0
        total_pixels = len(pixels)
        
        for pixel in pixels:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            # Check for red-ish colors (aggressive detection)
            if r > 120 and g < 100 and b < 100:
                red_pixels += 1
        
        # If red constitutes more than 10% of the image, force red detection
        if red_pixels > total_pixels * 0.1:
            print(f"[COLOR DETECTION] Aggressive red detection: {red_pixels}/{total_pixels} pixels are red-ish")
            return ["red"]
        
        # Simple k-means-like clustering for other colors
        from collections import defaultdict
        color_groups = []
        
        for pixel in pixels:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            # Aggressive quantization to reduce variations
            r_q = (r // 60) * 60
            g_q = (g // 60) * 60
            b_q = (b // 60) * 60
            
            found = False
            for i, (cr, cg, cb, count) in enumerate(color_groups):
                if abs(r_q - cr) < 80 and abs(g_q - cg) < 80 and abs(b_q - cb) < 80:
                    color_groups[i] = (
                        int((cr * count + r) / (count + 1)),
                        int((cg * count + g) / (count + 1)),
                        int((cb * count + b) / (count + 1)),
                        count + 1
                    )
                    found = True
                    break
            
            if not found and len(color_groups) < num_colors * 2:
                color_groups.append((r, g, b, 1))
        
        color_groups.sort(key=lambda x: x[3], reverse=True)
        
        # Get color names with confidence threshold
        colors = []
        for center in color_groups[:num_colors]:
            r, g, b, count = center
            confidence = count / total_pixels
            
            # Only include colors with sufficient confidence (>5% of image)
            if confidence > 0.05:
                color_name = get_color_name(r, g, b)
                if color_name not in colors:
                    colors.append(color_name)
        
        # If we detected colors, return them; otherwise return multi-color
        return colors if colors else ["multi-color"]
    except Exception as e:
        print(f"[WARNING] Color extraction error: {e}")
        return ["multi-color"]

def get_color_name(r: int, g: int, b: int) -> str:
    """Convert RGB to color name with aggressive red detection"""
    # Calculate brightness
    brightness = (r + g + b) / 3
    
    # AGGRESSIVE RED DETECTION - Check for red-ish colors first
    if r > 120 and g < 100 and b < 100:
        # Strong red detection
        if r > 150 and g < 80 and b < 80:
            return "red"
        # Lighter red/orange-ish
        elif g > 60:
            return "orange"
        else:
            return "red"
    
    # Black and white
    if brightness < 50:
        return "black"
    if brightness > 200 and abs(r - g) < 30 and abs(g - b) < 30:
        return "white"
    
    # Calculate dominant channel
    max_val = max(r, g, b)
    
    # Gray
    if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
        return "gray"
    
    # Primary and secondary colors (only if not red-ish)
    if r > g + 30 and r > b + 30:
        if g > b + 20:
            return "orange"
        return "red"
    elif g > r + 30 and g > b + 30:
        return "green"
    elif b > r + 30 and b > g + 30:
        if b > 180 and r > 100:
            return "purple"
        return "blue"
    elif r > 100 and g > 100 and b < 100:
        return "yellow"
    elif r > 100 and b > 100:
        return "pink" if brightness > 150 else "purple"
    elif g > 100 and b > 100:
        return "cyan"
    elif r > 80 and g > 60 and b < 70:
        return "brown"
    
    # Final fallback - if still has red-ish tint, call it red
    if r > 100 and g < 90 and b < 90:
        return "red"
    
    return "multi-color"

def analyze_image_with_clip(image: Image.Image) -> dict:
    """Use CLIP to analyze the image and determine product type"""
    try:
        # Product categories to test
        categories = [
            "headphones", "wireless headphones", "earbuds", "gaming headset",
            "smartphone", "mobile phone", "laptop", "tablet", "computer",
            "watch", "smartwatch", "fitness tracker",
            "shoes", "sneakers", "boots", "sandals",
            "bag", "backpack", "handbag", "purse", "luggage",
            "clothing", "shirt", "dress", "jacket", "pants",
            "camera", "DSLR camera", "action camera",
            "book", "notebook", "magazine",
            "furniture", "chair", "table", "desk",
            "toy", "action figure", "doll",
            "bottle", "water bottle", "thermos",
            "sunglasses", "eyeglasses",
            "jewelry", "necklace", "bracelet", "ring",
            "kitchen appliance", "blender", "coffee maker",
            "home decor", "lamp", "vase", "picture frame"
        ]
        
        # Create text prompts
        text_prompts = [f"a photo of {cat}" for cat in categories]
        
        # Encode image
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Encode text
        text_tokens = tokenizer(text_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T).squeeze(0)
            
        # Get top 5 matches
        top5_idx = similarity.topk(5).indices.cpu().numpy()
        top5_scores = similarity.topk(5).values.cpu().numpy()
        
        detected_categories = [(categories[idx], float(score)) for idx, score in zip(top5_idx, top5_scores)]
        
        # Get primary category
        primary_category = detected_categories[0][0]
        confidence = detected_categories[0][1]
        
        # Determine broader category
        broad_category = categorize_product(primary_category)
        
        return {
            "primary_category": primary_category,
            "confidence": confidence,
            "broad_category": broad_category,
            "top_matches": detected_categories[:3],
            "likely_product": confidence > 0.25
        }
        
    except Exception as e:
        print(f"[WARNING] CLIP analysis error: {e}")
        return {
            "primary_category": "product",
            "confidence": 0.0,
            "broad_category": "general",
            "top_matches": [],
            "likely_product": True
        }

def categorize_product(category: str) -> str:
    """Map specific category to broader category"""
    category_map = {
        "electronics": ["headphones", "earbuds", "smartphone", "phone", "laptop", "tablet", 
                       "computer", "camera", "gaming", "smartwatch", "fitness tracker"],
        "fashion": ["shoes", "sneakers", "boots", "clothing", "shirt", "dress", "jacket", 
                   "pants", "sandals"],
        "accessories": ["bag", "backpack", "handbag", "purse", "watch", "sunglasses", 
                       "eyeglasses", "jewelry", "necklace", "bracelet", "ring"],
        "home": ["furniture", "chair", "table", "lamp", "vase", "decor", "kitchen", 
                "appliance", "blender", "coffee maker"],
        "other": ["book", "notebook", "toy", "bottle", "magazine", "luggage"]
    }
    
    category_lower = category.lower()
    for broad_cat, keywords in category_map.items():
        if any(keyword in category_lower for keyword in keywords):
            return broad_cat
    
    return "general"

def extract_clip_embedding(image_bytes: bytes) -> List[float]:
    """Extract CLIP embedding from image bytes"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def filter_products_by_color(products: List[dict], target_colors: List[str]) -> List[dict]:
    """Filter products based on color similarity to target colors with strict matching"""
    if not target_colors or not products:
        return products
    
    print(f"[COLOR FILTER] Filtering {len(products)} products by colors: {target_colors}")
    
    filtered_products = []
    
    for product in products:
        title = product.get('title', '').lower()
        description = product.get('description', '').lower()
        combined_text = f"{title} {description}"
        
        # Check if any target color appears in the product text
        color_match = False
        match_count = 0
        
        for target_color in target_colors:
            # Check for exact color match
            if target_color in combined_text:
                match_count += 1
                continue
            
            # Check for color variations (e.g., "red" matches "crimson", "maroon")
            color_variations = get_color_variations(target_color)
            for variation in color_variations:
                if variation in combined_text:
                    match_count += 1
                    break
        
        # STRICT FILTERING: Only include if we match ALL detected colors
        # This prevents white dresses from appearing when searching for red
        if len(target_colors) == 1:
            # Single color search - require at least one match
            color_match = (match_count > 0)
        else:
            # Multiple colors detected - require matching at least 50% of them
            required_matches = max(1, len(target_colors) // 2)
            color_match = (match_count >= required_matches)
        
        # If we have sufficient color matches, include the product
        if color_match:
            filtered_products.append(product)
    
    print(f"[COLOR FILTER] Found {len(filtered_products)} products matching target colors")
    
    # If no products matched colors, return empty list instead of original
    # This prevents showing irrelevant results when color was specifically requested
    return filtered_products

def get_color_variations(color: str) -> List[str]:
    """Get variations of a color name"""
    color_variations = {
        "red": ["crimson", "maroon", "burgundy", "cherry", "scarlet", "ruby"],
        "blue": ["navy", "azure", "cobalt", "sapphire", "teal", "turquoise"],
        "green": ["emerald", "forest", "olive", "lime", "mint", "jade"],
        "yellow": ["gold", "mustard", "amber", "lemon", "cream"],
        "purple": ["violet", "lavender", "magenta", "plum", "lilac"],
        "orange": ["coral", "peach", "apricot", "tangerine"],
        "pink": ["rose", "blush", "fuchsia", "hot pink", "salmon"],
        "brown": ["chocolate", "tan", "beige", "caramel", "coffee"],
        "black": ["charcoal", "dark", "jet black"],
        "white": ["ivory", "cream", "off-white", "pearl", "snow"],
        "gray": ["grey", "silver", "charcoal", "ash", "slate"]
    }
    
    return color_variations.get(color.lower(), [])

async def fetch_from_google_custom_search(query: str, page: int = 1, max_retries: int = 3) -> List[dict]:
    """
    Fetch products from Google Custom Search API with rate limiting, caching, and retry logic
    """
    print(f"[SEARCH] Searching: '{query}' (page {page})")
    
    # Check cache first
    cached_result = await search_cache.get(query, page)
    if cached_result is not None:
        return cached_result
    
    products = []
    api_key = api_key_manager.get_current_key()
    
    if not api_key:
        print("[ERROR] No Google API key available")
        return products
    
    # Apply rate limiting
    await rate_limiter.wait_if_needed(api_key)
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        
        # Calculate start index for pagination
        start_index = ((page - 1) * 10) + 1
        
        # Enhanced search parameters for better product discovery
        params = {
            "q": query,
            "key": api_key,
            "cx": os.getenv("GOOGLE_CX"),
            "searchType": "image",
            "num": 10,
            "start": start_index,
            "imgSize": "large",
            "safe": "active",
            "fileType": "jpg,png",
            "imgType": "photo",
            # Add additional parameters for better product search
            "sort": "date",  # Prefer recent results
            "dateRestrict": "m6"  # Results from last 6 months
        }
        
        # Add site-specific searches for better platform coverage
        site_specific_queries = []
        if "flipkart" in query.lower():
            site_specific_queries.append(f"site:flipkart.com {query.replace('Flipkart', '').strip()}")
        if "meesho" in query.lower():
            site_specific_queries.append(f"site:meesho.com {query.replace('Meesho', '').strip()}")
        if "amazon" in query.lower():
            site_specific_queries.append(f"site:amazon.in OR site:amazon.com {query.replace('Amazon', '').strip()}")
        if "myntra" in query.lower():
            site_specific_queries.append(f"site:myntra.com {query.replace('Myntra', '').strip()}")
        if "ajio" in query.lower():
            site_specific_queries.append(f"site:ajio.com {query.replace('Ajio', '').strip()}")
        
        # Try multiple search variations for better coverage
        search_variations = [query] + site_specific_queries
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for search_query in search_variations:
                if len(products) >= 10:  # Limit per variation
                    break
                
                current_params = params.copy()
                current_params["q"] = search_query
                
                # Implement exponential backoff for retries
                for attempt in range(max_retries):
                    try:
                        response = await client.get(url, params=current_params)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            if data.get('items'):
                                for item in data['items']:
                                    try:
                                        if len(products) >= 10:  # Limit per variation
                                            break
                                            
                                        title = item.get('title', 'Product')
                                        image_url = item.get('link')
                                        
                                        if not image_url:
                                            continue
                                        
                                        # Skip low-quality images
                                        if any(skip_term in image_url.lower() for skip_term in ['placeholder', 'no-image', 'default']):
                                            continue
                                        
                                        # Get context
                                        context = item.get('image', {})
                                        link = context.get('contextLink', image_url)
                                        snippet = item.get('snippet', title)
                                        
                                        # Try to extract price from snippet or title
                                        price = extract_price_from_text(f"{title} {snippet}")
                                        
                                        # Detect actual source from URL
                                        source = detect_source_from_url(link)
                                        
                                        product = {
                                            'title': clean_title(title),
                                            'image_url': image_url,
                                            'price': price,
                                            'description': snippet[:200],
                                            'link': link,
                                            'source': source
                                        }
                                        products.append(product)
                                        
                                    except Exception as e:
                                        print(f"[WARNING] Error parsing item: {e}")
                                        continue
                                
                                print(f"  [OK] Found {len(products)} products from '{search_query}'")
                                break  # Success, move to next variation
                                
                            else:
                                print(f"  [INFO] No items for '{search_query}'")
                                break
                                
                        elif response.status_code == 429:
                            # Rate limit hit - try rotating API key first
                            print(f"  [RATE LIMIT] Hit rate limit for current API key")
                            
                            # Try to rotate to next API key
                            new_key = api_key_manager.rotate_key()
                            if new_key and new_key != api_key:
                                api_key = new_key
                                current_params["key"] = api_key
                                print(f"  [RATE LIMIT] Rotated to new API key, retrying...")
                                # Apply rate limiting for new key
                                await rate_limiter.wait_if_needed(api_key)
                                continue
                            else:
                                # No more keys available, implement exponential backoff
                                wait_time = (2 ** attempt) + (0.5 * attempt)  # 1s, 3s, 7s
                                print(f"  [RATE LIMIT] No more API keys, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                            
                        else:
                            print(f"  [ERROR] API error: {response.status_code} for '{search_query}'")
                            break
                            
                    except Exception as e:
                        print(f"  [ERROR] Request failed: {e} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            break
                
                # Delay between search variations to avoid overwhelming the API
                if search_variations.index(search_query) < len(search_variations) - 1:
                    await asyncio.sleep(1.0)  # Increased delay
                    
        # Cache the results
        await search_cache.set(query, page, products)
        
    except Exception as e:
        print(f"  [ERROR] Search error: {e}")
    
    print(f"[SEARCH] Completed search for '{query}' page {page}: found {len(products)} products")
    return products

def extract_price_from_text(text: str) -> str:
    """Extract price from text"""
    import re
    
    # Look for price patterns
    price_patterns = [
        r'\$\s*\d+\.?\d*',
        r'₹\s*\d+\.?\d*',
        r'€\s*\d+\.?\d*',
        r'£\s*\d+\.?\d*',
        r'\d+\.?\d*\s*(?:USD|INR|EUR|GBP)',
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return "Check website"

def clean_title(title: str) -> str:
    """Clean product title"""
    # Remove common noise
    noise_patterns = [
        r'\|\s*.*$',  # Remove everything after |
        r'\-\s*.*$',  # Remove everything after -
        r'\d{3,}x\d{3,}',  # Remove image dimensions
    ]
    
    for pattern in noise_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    return title.strip()[:150]

def detect_source_from_url(url: str) -> str:
    """Detect the e-commerce platform from URL"""
    if not url:
        return "Unknown"
    
    url_lower = url.lower()
    
    # Platform detection based on domain
    if 'amazon.' in url_lower:
        return 'Amazon'
    elif 'flipkart.' in url_lower:
        return 'Flipkart'
    elif 'meesho.' in url_lower:
        return 'Meesho'
    elif 'myntra.' in url_lower:
        return 'Myntra'
    elif 'ajio.' in url_lower:
        return 'Ajio'
    elif 'nykaa.' in url_lower:
        return 'Nykaa'
    elif 'tatacliq.' in url_lower:
        return 'Tata Cliq'
    elif 'reliancedigital.' in url_lower:
        return 'Reliance Digital'
    elif 'croma.' in url_lower:
        return 'Croma'
    elif 'vijaysales.' in url_lower:
        return 'Vijay Sales'
    elif 'snapdeal.' in url_lower:
        return 'Snapdeal'
    elif 'shopclues.' in url_lower:
        return 'Shopclues'
    elif 'paytmmall.' in url_lower:
        return 'Paytm Mall'
    elif 'firstcry.' in url_lower:
        return 'FirstCry'
    elif 'jabong.' in url_lower or 'abof.' in url_lower:
        return 'Fashion Store'
    elif 'bestbuy.' in url_lower:
        return 'Best Buy'
    elif 'walmart.' in url_lower:
        return 'Walmart'
    elif 'target.' in url_lower:
        return 'Target'
    elif 'ebay.' in url_lower:
        return 'eBay'
    elif 'aliexpress.' in url_lower:
        return 'AliExpress'
    elif 'shopify.' in url_lower:
        return 'Shopify Store'
    elif 'google.' in url_lower and ('shopping' in url_lower or 'products' in url_lower):
        return 'Google Shopping'
    else:
        # Try to extract domain name for unknown sources
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if '.' in domain:
                domain = domain.split('.')[0]
            return domain.capitalize() if domain else 'Unknown'
        except:
            return 'Unknown'

def generate_search_queries(category_info: dict, colors: List[str], broad_category: str) -> List[str]:
    """Generate targeted search queries based on detected product"""
    primary_category = category_info.get("primary_category", "product")
    
    queries = []
    
    # Add primary category queries
    queries.append(f"{primary_category} product")
    queries.append(f"{primary_category} buy online")
    queries.append(f"{primary_category} shopping")
    
    # Add color variations
    for color in colors[:2]:
        if color not in ["multi-color", "mixed"]:
            queries.append(f"{color} {primary_category}")
            queries.append(f"{primary_category} {color}")
    
    # Add platform-specific queries for major e-commerce sites
    platform_queries = [
        f"{primary_category} Amazon",
        f"{primary_category} Flipkart",
        f"{primary_category} Meesho", 
        f"{primary_category} Google Shopping",
        f"{primary_category} Myntra",
        f"{primary_category} Ajio",
        f"{primary_category} Nykaa",
        f"{primary_category} Tata Cliq",
        f"{primary_category} Reliance Digital",
        f"{primary_category} Croma",
        f"{primary_category} Vijay Sales"
    ]
    
    # Add category-specific queries
    if broad_category == "electronics":
        queries.extend([
            f"{primary_category} Best Buy",
            f"wireless {primary_category}",
            f"bluetooth {primary_category}",
            f"{primary_category} tech",
            f"{primary_category} gadget"
        ])
        # Add electronics-specific platforms
        platform_queries.extend([
            f"{primary_category} electronic store",
            f"{primary_category} tech store"
        ])
    elif broad_category == "fashion":
        queries.extend([
            f"{primary_category} fashion",
            f"{primary_category} style",
            f"{primary_category} clothing",
            f"{primary_category} apparel"
        ])
        # Add fashion-specific platforms
        platform_queries.extend([
            f"{primary_category} fashion store",
            f"{primary_category} boutique"
        ])
    elif broad_category == "accessories":
        queries.extend([
            f"{primary_category} accessories",
            f"stylish {primary_category}",
            f"designer {primary_category}",
            f"{primary_category} collection"
        ])
    
    # Add platform-specific queries (shuffle to avoid bias)
    import random
    shuffled_platforms = platform_queries.copy()
    random.shuffle(shuffled_platforms)
    queries.extend(shuffled_platforms)
    
    # Add generic shopping queries
    queries.extend([
        f"{primary_category} 2024",
        f"best {primary_category}",
        f"popular {primary_category}",
        f"{primary_category} online",
        f"{primary_category} store",
        f"buy {primary_category}",
        f"{primary_category} deal",
        f"{primary_category} offer"
    ])
    
    # Remove duplicates and limit
    seen = set()
    unique_queries = []
    for query in queries:
        if query.lower() not in seen:
            seen.add(query.lower())
            unique_queries.append(query)
    
    return unique_queries[:20]  # Increased limit to 20 for more variety

async def index_products(products: List[dict]) -> int:
    """Index products in Qdrant with CLIP embeddings"""
    print(f"\n[INDEX] Indexing {len(products)} products...")
    points = []
    failed = 0
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for idx, product in enumerate(products):
            try:
                if (idx + 1) % 5 == 0:
                    print(f"  Processing {idx+1}/{len(products)}...")
                
                # Download image
                response = await client.get(product["image_url"])
                if response.status_code == 200:
                    # Extract embedding
                    embedding = extract_clip_embedding(response.content)
                    
                    # Create point
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=product
                    )
                    points.append(point)
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                if idx < 5:  # Only print first few errors
                    print(f"  [WARNING] Error on product {idx+1}: {str(e)[:50]}")
                continue
    
    if points:
        print(f"  Uploading {len(points)} vectors to Qdrant...")
        # Upload in batches
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        print(f"  [OK] Upload complete!")
    
    print(f"  Success: {len(points)} | Failed: {failed}")
    return len(points)

@app.post("/search", response_model=SearchResponse)
async def search_similar_products(image: UploadFile = File(...)):
    """
    Upload an image and find visually similar products
    """
    try:
        # Clear existing database
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print("[OK] Database cleared")
        except Exception as e:
            print(f"[WARNING] Database clear error: {e}")
        
        # Read image
        image_bytes = await image.read()
        
        print("\n" + "="*70)
        print("[VISUAL PRODUCT SEARCH]")
        print("="*70)
        
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Step 1: Analyze image with CLIP
        print("\n[ANALYZE] Analyzing image with CLIP...")
        category_info = analyze_image_with_clip(img)
        colors = extract_dominant_colors(img)
        
        print(f"  Detected: {category_info['primary_category']}")
        print(f"  Confidence: {category_info['confidence']:.2%}")
        print(f"  Category: {category_info['broad_category']}")
        print(f"  Colors: {', '.join(colors)}")
        
        if len(category_info['top_matches']) > 1:
            print(f"  Also matches: {', '.join([m[0] for m in category_info['top_matches'][1:3]])}")
        
        # Step 2: Generate search queries
        print(f"\n[QUERY] Generating search queries...")
        search_queries = generate_search_queries(
            category_info, 
            colors, 
            category_info['broad_category']
        )
        print(f"  Generated {len(search_queries)} queries")
        
        # Step 3: Search Google
        print(f"\n[GOOGLE] Searching Google Custom Search API...")
        all_products = []
        max_products = 100
        
        for i, query in enumerate(search_queries):
            if len(all_products) >= max_products:
                break
            
            # Search with pagination (2 pages per query)
            for page in [1, 2]:
                if len(all_products) >= max_products:
                    break
                    
                products = await fetch_from_google_custom_search(query, page)
                all_products.extend(products)
                
                # Rate limit protection
                if page == 1 and products:
                    await asyncio.sleep(0.5)
        
        # Remove duplicates
        seen_urls = set()
        unique_products = []
        for product in all_products:
            url = product['image_url']
            if url not in seen_urls:
                seen_urls.add(url)
                unique_products.append(product)
        
        print(f"\n[STATS] Total unique products found: {len(unique_products)}")
        
        if not unique_products:
            raise HTTPException(
                status_code=404,
                detail="No products found. Please check your Google Custom Search API configuration."
            )
        
        # Step 4: Index products
        query_embedding = extract_clip_embedding(image_bytes)
        indexed_count = await index_products(unique_products)
        
        if indexed_count == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to index products"
            )
        
        # Step 5: Find similar products
        print(f"\n[MATCH] Finding visually similar products...")
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=30
        )
        
        # Filter and rank results
        print(f"  Found {len(search_results)} candidates")
        
        # Dynamic threshold based on category
        min_threshold = 0.60 if category_info['broad_category'] == 'electronics' else 0.65
        
        filtered_results = []
        seen_titles = set()
        
        for hit in search_results:
            if len(filtered_results) >= 20:
                break
                
            if hit.score >= min_threshold:
                title_lower = hit.payload['title'].lower()
                if title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    filtered_results.append(hit)
        
        print(f"  Filtered to {len(filtered_results)} high-quality matches")
        
        # Apply color filtering if colors were detected
        if colors:
            print(f"[COLOR] Applying color filter for: {colors}")
            # Convert filtered results to products list for color filtering
            products_for_color_filter = [hit.payload for hit in filtered_results]
            color_filtered_products = filter_products_by_color(products_for_color_filter, colors)
            
            # Create a mapping of products that passed color filter
            color_filtered_urls = {product['image_url'] for product in color_filtered_products}
            
            # Filter the original search results based on color filtering
            final_filtered_results = []
            for hit in filtered_results:
                if hit.payload['image_url'] in color_filtered_urls:
                    final_filtered_results.append(hit)
            
            print(f"[COLOR] After color filtering: {len(final_filtered_results)} products")
            
            # If we have color-filtered results, use them; otherwise keep original
            if final_filtered_results:
                filtered_results = final_filtered_results
        
        # Format results
        results = []
        sources = set()
        
        for hit in filtered_results:
            sources.add(hit.payload.get("source", "Unknown"))
            results.append(ProductResult(
                title=hit.payload["title"],
                image_url=hit.payload["image_url"],
                price=hit.payload.get("price"),
                description=hit.payload.get("description"),
                link=hit.payload["link"],
                similarity=hit.score,
                source=hit.payload.get("source", "Unknown")
            ))
        
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        print("\n" + "="*70)
        print(f"[COMPLETE] SEARCH COMPLETE: {len(results)} results")
        print("="*70 + "\n")
        
        return SearchResponse(
            results=results,
            total_scraped=len(unique_products),
            sources=list(sources),
            detected_category=category_info['primary_category'],
            detected_attributes={
                "colors": colors,
                "broad_category": category_info['broad_category'],
                "confidence": category_info['confidence']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "CLIP ViT-B/32",
        "device": device,
        "apis_configured": ["Google Custom Search API"] if os.getenv("GOOGLE_API_KEY") else [],
    }

# Collection endpoints
@app.post("/collection/add")
async def add_to_collection(image_url: str, user_id: str = "anonymous"):
    """Add an image to user's collection"""
    try:
        # Check if image already exists in collection
        existing = await collection_collection.find_one({
            "user_id": user_id,
            "image_url": image_url
        })
        
        if existing:
            return {"message": "Image already in collection", "collection_id": str(existing["_id"])}
        
        # Add new collection item
        collection_item = {
            "user_id": user_id,
            "image_url": image_url,
            "created_at": datetime.now()
        }
        
        result = await collection_collection.insert_one(collection_item)
        
        return {
            "message": "Image added to collection",
            "collection_id": str(result.inserted_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add to collection: {str(e)}")

@app.delete("/collection/remove/{collection_id}")
async def remove_from_collection(collection_id: str, user_id: str = "anonymous"):
    """Remove an image from user's collection"""
    try:
        result = await collection_collection.delete_one({
            "_id": ObjectId(collection_id),
            "user_id": user_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Collection item not found")
        
        return {"message": "Image removed from collection"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove from collection: {str(e)}")

@app.get("/collection/list")
async def get_user_collection(user_id: str = "anonymous") -> CollectionResponse:
    """Get user's image collection"""
    try:
        cursor = collection_collection.find({"user_id": user_id}).sort("created_at", -1)
        items = []
        
        async for doc in cursor:
            items.append(CollectionItem(
                image_url=doc["image_url"],
                collection_id=str(doc["_id"]),
                created_at=doc["created_at"]
            ))
        
        return CollectionResponse(items=items, total=len(items))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection: {str(e)}")

@app.delete("/collection/clear")
async def clear_user_collection(user_id: str = "anonymous"):
    """Clear user's entire collection"""
    try:
        result = await collection_collection.delete_many({"user_id": user_id})
        return {"message": f"Collection cleared. {result.deleted_count} items removed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")

@app.delete("/clear-database")
async def clear_database():
    """Clear all vectors from Qdrant"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        return {"status": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import re
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")