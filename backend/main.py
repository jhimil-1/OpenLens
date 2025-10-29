from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
import tempfile
import cv2
import mediapipe as mp
from gradio_client import Client, handle_file
import re

load_dotenv()

# Rate limiting and caching system
class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 100):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self, api_key: str):
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            self.requests[api_key] = [req_time for req_time in self.requests[api_key] if req_time > minute_ago]
            
            if len(self.requests[api_key]) >= self.max_requests:
                oldest_request = min(self.requests[api_key])
                wait_time = (oldest_request + 60) - now
                if wait_time > 0:
                    print(f"[RATE LIMIT] Waiting {wait_time:.1f}s to avoid API limits")
                    await asyncio.sleep(wait_time)
            
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
rate_limiter = RateLimiter(max_requests_per_minute=90)
search_cache = SearchCache(ttl_hours=24)

class APIKeyManager:
    def __init__(self):
        self.keys = []
        self.current_index = 0
        self.load_keys()
    
    def load_keys(self):
        primary_key = os.getenv("GOOGLE_API_KEY")
        if primary_key:
            self.keys.append(primary_key)
        
        backup_keys = [
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3"),
            "AIzaSyBn8XLzv_GM18y8MLoVUkHT-F9yQIoeaH0",
            "AIzaSyAmkVShutgc_MBRFSH43WY7o8SMqlhntsc"
        ]
        
        for key in backup_keys:
            if key and key not in self.keys:
                self.keys.append(key)
        
        print(f"[API KEYS] Loaded {len(self.keys)} API keys")
    
    def get_current_key(self) -> str:
        if not self.keys:
            return None
        return self.keys[self.current_index]
    
    def rotate_key(self) -> str:
        if not self.keys:
            return None
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"[API KEYS] Rotated to key {self.current_index + 1}/{len(self.keys)}")
        return self.keys[self.current_index]
    
    def get_all_keys(self) -> List[str]:
        return self.keys.copy()

# Initialize API key manager
api_key_manager = APIKeyManager()

# Initialize MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db_name = os.getenv("MONGODB_DB_NAME", "openlens")
collection_name = os.getenv("MONGODB_COLLECTION_NAME", "image_collection")

db = mongo_client[db_name]
collection_collection = db[collection_name]

print(f"[MONGODB] Connected to database: {db_name}, collection: {collection_name}")

app = FastAPI(title="Visual Product Search & Virtual Try-On API")

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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
mp_pose_landmark = mp_pose.PoseLandmark

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

# ────────────────────────────────
# IMPROVED HELPER FUNCTIONS
# ────────────────────────────────

def detect_pose(image):
    """Detect pose landmarks from an image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    keypoints = {}

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        height, width, _ = image.shape

        for name, index in {
            'left_shoulder': mp_pose_landmark.LEFT_SHOULDER,
            'right_shoulder': mp_pose_landmark.RIGHT_SHOULDER,
            'left_hip': mp_pose_landmark.LEFT_HIP,
            'right_hip': mp_pose_landmark.RIGHT_HIP,
        }.items():
            lm = result.pose_landmarks.landmark[index]
            keypoints[name] = (int(lm.x * width), int(lm.y * height))

    return keypoints

def process_tryon_image(human_img_path: str, garm_img_path: str):
    """Call the Gradio client (Leffa model) to perform virtual try-on."""
    try:
        client = Client("franciszzj/Leffa")

        result = client.predict(
            src_image_path=handle_file(human_img_path),
            ref_image_path=handle_file(garm_img_path),
            ref_acceleration=False,
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,
            api_name="/leffa_predict_vt",
        )

        generated_image_path = result[0]
        return generated_image_path
    except Exception as e:
        print(f"[ERROR] Virtual Try-On processing failed: {e}")
        raise

def extract_dominant_colors(image: Image.Image, num_colors: int = 2) -> List[str]:
    """Extract garment colors using HSV with central focus and background suppression.
    Designed to avoid picking white/gray backgrounds for items like a blue dress.
    """
    try:
        image = image.convert("RGB")
        image = image.resize((128, 128))

        img = np.array(image)
        h, w, _ = img.shape

        # Central ellipse mask → prioritize the garment region
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h * 0.35, w * 0.35
        center_mask = (((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2)) <= 1.0

        # HSV conversion
        if 'cv2' in globals() and cv2 is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_scale = 180.0
        else:
            rgb = img.astype(np.float32) / 255.0
            maxc = rgb.max(axis=2)
            minc = rgb.min(axis=2)
            v = maxc
            s = np.where(v == 0, 0, (maxc - minc) / (v + 1e-6))
            # approximate OpenCV-like HSV (H unused in fallback; we build from RGB later if needed)
            hsv = np.stack([np.zeros_like(v), s * 255.0, v * 255.0], axis=2)
            hue_scale = 180.0

        H = hsv[:, :, 0]
        S = hsv[:, :, 1] / 255.0
        V = hsv[:, :, 2] / 255.0

        # First, decide if the central region is predominantly dark or bright
        center_V = V[center_mask]
        if center_V.size > 0:
            dark_ratio = float(np.mean(center_V < 0.18))
            bright_ratio = float(np.mean(center_V > 0.92))
            if dark_ratio > 0.55:
                return ["black"]
            if bright_ratio > 0.55:
                return ["white"]

        # Keep well-saturated, mid-value pixels; drop near-white/near-black
        sat_mask = S > 0.28
        val_mask = (V > 0.12) & (V < 0.92)
        keep = center_mask & sat_mask & val_mask

        # If no pixels, relax to whole image with same gates; if still empty, decide by global brightness
        if not keep.any():
            keep = sat_mask & val_mask
        if not keep.any():
            global_dark = float(np.mean(V < 0.18))
            global_bright = float(np.mean(V > 0.92))
            if global_dark > 0.55:
                return ["black"]
            if global_bright > 0.55:
                return ["white"]
            return ["multi-color"]

        # Estimate border average color and drop pixels close to it (likely background)
        border = np.concatenate([img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]], axis=0)
        bg_mean = border.mean(axis=0)
        dist_bg = np.sqrt(((img - bg_mean) ** 2).sum(axis=2))
        keep = keep & (dist_bg > 35.0)

        if not keep.any():
            return ["multi-color"]

        hue = H[keep]
        # Histogram over hue to find dominant hues
        bins = np.linspace(0, hue_scale, 13, endpoint=True)
        hist, edges = np.histogram(hue, bins=bins)
        order = np.argsort(hist)[::-1]

        def hue_to_name(hval: float) -> str:
            # OpenCV hue [0,180]
            if hval < 10 or hval >= 170:
                return "red"
            if hval < 25:
                return "orange"
            if hval < 35:
                return "yellow"
            if hval < 85:
                return "green"
            if hval < 105:
                return "cyan"
            if hval < 140:
                return "blue"
            return "purple"

        colors: List[str] = []
        for idx in order[: max(1, num_colors + 1)]:
            center_h = (edges[idx] + edges[idx + 1]) / 2.0
            name = hue_to_name(center_h)
            if name not in colors:
                colors.append(name)
            if len(colors) >= num_colors:
                break

        # Strong red safeguard: if red present make it primary
        if 'red' in colors and colors[0] != 'red':
            colors.remove('red')
            colors.insert(0, 'red')

        return colors if colors else ["multi-color"]
    except Exception as e:
        print(f"[WARNING] Color extraction error: {e}")
        return ["multi-color"]

def get_color_name(r: int, g: int, b: int) -> str:
    """Convert RGB to color name with improved detection"""
    brightness = (r + g + b) / 3
    
    # More precise color detection
    if r > 150 and g < 100 and b < 100 and r > max(g, b) + 50:
        if g > 80:
            return "orange"
        elif r > 200 and g < 60 and b < 60:
            return "red"
        else:
            return "red"
    
    if brightness < 60:
        return "black"
    if brightness > 220 and abs(r - g) < 30 and abs(g - b) < 30:
        return "white"
    
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    
    # Gray detection
    if max_val - min_val < 30:
        if brightness < 150:
            return "gray"
        else:
            return "white"
    
    # Primary colors with better thresholds
    if r > g + 40 and r > b + 40:
        if g > b + 30:
            return "orange"
        elif r > 180:
            return "red"
        else:
            return "red"
    elif g > r + 40 and g > b + 40:
        if g > 150:
            return "green"
        else:
            return "dark green"
    elif b > r + 40 and b > g + 40:
        if b > 150 and r > 100:
            return "purple"
        elif b > 150:
            return "blue"
        else:
            return "dark blue"
    elif r > 150 and g > 150 and b < 100:
        return "yellow"
    elif r > 150 and b > 150 and g < 100:
        return "purple"
    elif g > 150 and b > 150 and r < 100:
        return "teal"
    elif r > 120 and g > 80 and b < 80:
        return "brown"
    
    return "multi-color"

def detect_gender_from_category(primary_category: str, all_categories: List[tuple]) -> str:
    """Detect gender from category names with improved logic"""
    male_indicators = ["men's", "men", "male", "boy's", "boy", "gentleman", "gents"]
    female_indicators = ["women's", "women", "female", "girl's", "girl", "lady", "ladies", "woman's", "feminine"]
    
    primary_lower = primary_category.lower()
    
    # Strong female indicators in primary category
    for indicator in female_indicators:
        if indicator in primary_lower:
            return "women"
    
    # Strong male indicators in primary category
    for indicator in male_indicators:
        if indicator in primary_lower:
            return "men"
    
    # Check all detected categories with confidence scores
    male_score = 0
    female_score = 0
    
    for category, score in all_categories:
        category_lower = category.lower()
        
        for indicator in male_indicators:
            if indicator in category_lower:
                male_score += score
        
        for indicator in female_indicators:
            if indicator in category_lower:
                female_score += score
    
    print(f"  [GENDER DEBUG] Male score: {male_score:.3f}, Female score: {female_score:.3f}")
    
    # Strong gender preference
    if female_score > 0.4 and male_score == 0:
        return "women"
    elif male_score > 0.4 and female_score == 0:
        return "men"
    elif female_score > male_score + 0.15:
        return "women"
    elif male_score > female_score + 0.15:
        return "men"
    
    # Category-based fallback (strong female indicators)
    strongly_female_categories = ["dress", "skirt", "blouse", "leggings", "maxi dress", "evening dress", "party dress"]
    strongly_male_categories = ["suit", "polo shirt", "blazer", "formal shirt", "tie"]
    
    for category in strongly_female_categories:
        if category in primary_lower:
            return "women"
    
    for category in strongly_male_categories:
        if category in primary_lower:
            return "men"
    
    return "unisex"

def analyze_image_with_clip(image: Image.Image) -> dict:
    """Use CLIP to analyze the image and determine product type with improved gender detection"""
    try:
        # Enhanced categories with better gender-specific items and more variety
        categories = [
            # Women's clothing (more specific and varied)
            "women's dress", "women's evening dress", "women's summer dress", "women's party dress",
            "women's maxi dress", "women's casual dress", "women's black dress", "women's white dress",
            "women's cocktail dress", "women's formal dress", "women's wedding dress",
            "women's skirt", "women's mini skirt", "women's long skirt", "women's pleated skirt",
            "women's top", "women's blouse", "women's t-shirt", "women's shirt", "women's crop top",
            "women's jeans", "women's pants", "women's trousers", "women's leggings", "women's yoga pants",
            "women's jacket", "women's coat", "women's sweater", "women's hoodie", "women's cardigan",
            "women's shorts", "women's jumpsuit", "women's romper", "women's bikini",
            "women's handbag", "women's purse", "women's heels", "women's sandals",
            
            # Men's clothing  
            "men's shirt", "men's t-shirt", "men's polo shirt", "men's formal shirt",
            "men's jeans", "men's pants", "men's trousers", "men's chinos", "men's cargo pants",
            "men's jacket", "men's blazer", "men's suit", "men's coat", "men's leather jacket",
            "men's shorts", "men's sweater", "men's hoodie", "men's tracksuit",
            "men's watch", "men's shoes", "men's sneakers", "men's boots",
            
            # Gender neutral (less specific)
            "dress", "skirt", "top", "blouse", "shirt", "t-shirt",
            "jeans", "pants", "trousers", "jacket", "coat", "sweater",
            
            # Accessories
            "shoes", "handbag", "jewelry", "watch", "sunglasses", "backpack",
            
            # Electronics
            "smartphone", "laptop", "headphones", "camera", "tablet"
        ]
        
        text_prompts = [f"a photo of {cat}" for cat in categories]
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_tokens = tokenizer(text_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).squeeze(0)
            
        top10_idx = similarity.topk(10).indices.cpu().numpy()
        top10_scores = similarity.topk(10).values.cpu().numpy()
        
        detected_categories = [(categories[idx], float(score)) for idx, score in zip(top10_idx, top10_scores)]
        
        primary_category = detected_categories[0][0]
        confidence = detected_categories[0][1]
        
        # Enhanced gender detection with more context
        gender = detect_gender_from_category(primary_category, detected_categories)
        broad_category = categorize_product(primary_category, gender)
        
        # Debug information
        print(f"  [DEBUG] Top categories:")
        for i, (cat, score) in enumerate(detected_categories[:5]):
            print(f"    {i+1}. {cat} ({score:.3f})")
        
        return {
            "primary_category": primary_category,
            "confidence": confidence,
            "broad_category": broad_category,
            "gender": gender,
            "top_matches": detected_categories[:5],
            "likely_product": confidence > 0.15
        }
        
    except Exception as e:
        print(f"[WARNING] CLIP analysis error: {e}")
        return {
            "primary_category": "product",
            "confidence": 0.0,
            "broad_category": "general",
            "gender": "unisex",
            "top_matches": [],
            "likely_product": True
        }

def categorize_product(category: str, gender: str) -> str:
    """Map specific category to broader category with gender consideration"""
    category_lower = category.lower()
    
    # Gender-specific categories
    if gender == "women":
        if any(term in category_lower for term in ["dress", "skirt", "blouse", "leggings", "jumpsuit", "romper"]):
            return "women_fashion"
        elif any(term in category_lower for term in ["women", "woman", "lady", "female"]):
            return "women_fashion"
    
    elif gender == "men":
        if any(term in category_lower for term in ["suit", "polo", "blazer", "formal shirt", "tie"]):
            return "men_fashion"
        elif any(term in category_lower for term in ["men", "man", "gentleman", "male"]):
            return "men_fashion"
    
    # General category mapping
    category_map = {
        "electronics": ["headphones", "smartphone", "laptop", "tablet", "camera", 
                       "smartwatch", "phone", "computer", "gaming"],
        "shoes": ["shoes", "sneakers", "boots", "sandals", "heels", "footwear"],
        "accessories": ["handbag", "backpack", "watch", "sunglasses", "jewelry", "purse"],
        "home": ["furniture", "home decor", "lamp", "vase"],
        "other": ["book", "toy", "bottle", "kitchen"]
    }
    
    for broad_cat, keywords in category_map.items():
        if any(keyword in category_lower for keyword in keywords):
            return broad_cat
    
    # Fallback to general fashion if it contains clothing terms
    clothing_terms = ["jeans", "pants", "shirt", "dress", "jacket", "skirt", "top", "sweater", "hoodie"]
    if any(term in category_lower for term in clothing_terms):
        if gender == "women":
            return "women_fashion"
        elif gender == "men":
            return "men_fashion"
        else:
            return "general_fashion"
    
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

def filter_products_by_attributes(products: List[dict], target_colors: List[str], gender: str, category: str) -> List[dict]:
    """Filter products based on color, gender, and category with improved matching"""
    if not products:
        return products
    
    print(f"[FILTER] Filtering {len(products)} products by: colors={target_colors}, gender={gender}, category={category}")
    
    filtered_products = []
    
    for product in products:
        title = product.get('title', '').lower()
        description = product.get('description', '').lower()
        combined_text = f"{title} {description}"
        
        # Color matching (relaxed for black/white)
        color_match = True
        if target_colors and target_colors != ["multi-color"]:
            color_match_count = 0
            for target_color in target_colors:
                if target_color in combined_text:
                    color_match_count += 1
                    continue
                
                color_variations = get_color_variations(target_color)
                for variation in color_variations:
                    if variation in combined_text:
                        color_match_count += 1
                        break
            
            # Relaxed matching for essential colors
            if len(target_colors) == 1:
                color_match = (color_match_count > 0)
            else:
                required_matches = max(1, len(target_colors) // 2)
                color_match = (color_match_count >= required_matches)
        
        # Gender matching (STRICT)
        gender_match = True
        if gender != "unisex":
            male_terms = ["men", "men's", "male", "boy", "boy's", "guy", "gentleman", "gents"]
            female_terms = ["women", "women's", "female", "girl", "girl's", "lady", "ladies", "woman's"]
            
            if gender == "women":
                # STRICT: Remove products that are explicitly for men
                if any(term in combined_text for term in male_terms):
                    gender_match = False
                # Bonus: Keep products that are explicitly for women
                elif any(term in combined_text for term in female_terms):
                    gender_match = True
                # For neutral products, keep them
                else:
                    gender_match = True
                    
            elif gender == "men":
                # STRICT: Remove products that are explicitly for women
                if any(term in combined_text for term in female_terms):
                    gender_match = False
                # Bonus: Keep products that are explicitly for men
                elif any(term in combined_text for term in male_terms):
                    gender_match = True
                # For neutral products, keep them
                else:
                    gender_match = True
        
        # Category matching
        category_match = True
        if category and category != "general":
            category_terms = get_category_search_terms(category, gender)
            if category_terms:
                category_match = any(term in combined_text for term in category_terms)
        
        if color_match and gender_match and category_match:
            filtered_products.append(product)
    
    print(f"[FILTER] Found {len(filtered_products)} matching products")
    return filtered_products

def get_category_search_terms(category: str, gender: str) -> List[str]:
    """Get search terms for specific categories with gender context"""
    base_terms = {
        "women_fashion": ["dress", "skirt", "blouse", "top", "leggings", "jumpsuit"],
        "men_fashion": ["shirt", "polo", "jeans", "pants", "trousers", "jacket"],
        "general_fashion": ["dress", "shirt", "jeans", "pants", "jacket", "top"],
        "shoes": ["shoes", "sneakers", "boots", "sandals"],
        "electronics": ["phone", "laptop", "headphone", "camera", "tablet"]
    }
    
    terms = base_terms.get(category, [])
    
    # Add gender-specific terms if available
    if gender == "women" and category in ["women_fashion", "general_fashion"]:
        terms.extend(["dress", "skirt", "blouse", "leggings"])
    elif gender == "men" and category in ["men_fashion", "general_fashion"]:
        terms.extend(["shirt", "polo", "jeans", "trousers"])
    
    return list(set(terms))

def get_color_variations(color: str) -> List[str]:
    """Get variations of a color name"""
    color_variations = {
        "red": ["crimson", "maroon", "burgundy", "cherry", "scarlet", "ruby"],
        "blue": ["navy", "azure", "cobalt", "sapphire", "teal", "turquoise", "sky blue"],
        "green": ["emerald", "forest", "olive", "lime", "mint", "jade"],
        "yellow": ["gold", "mustard", "amber", "lemon", "cream"],
        "purple": ["violet", "lavender", "magenta", "plum", "lilac"],
        "orange": ["coral", "peach", "apricot", "tangerine"],
        "pink": ["rose", "blush", "fuchsia", "hot pink", "salmon"],
        "brown": ["chocolate", "tan", "beige", "caramel", "coffee"],
        "black": ["charcoal", "dark", "jet black", "ebony"],
        "white": ["ivory", "cream", "off-white", "pearl", "snow"],
        "gray": ["grey", "silver", "charcoal", "ash", "slate"]
    }
    
    return color_variations.get(color.lower(), [])

async def fetch_from_google_custom_search(query: str, page: int = 1, max_retries: int = 3) -> List[dict]:
    """Fetch products from Google Custom Search API with improved query handling"""
    print(f"[SEARCH] Searching: '{query}' (page {page})")
    
    cached_result = await search_cache.get(query, page)
    if cached_result is not None:
        return cached_result
    
    products = []
    api_key = api_key_manager.get_current_key()
    
    if not api_key:
        print("[ERROR] No Google API key available")
        return products
    
    await rate_limiter.wait_if_needed(api_key)
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        
        start_index = ((page - 1) * 10) + 1
        
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
            "imgType": "photo"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(url, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get('items'):
                            for item in data['items']:
                                try:
                                    title = item.get('title', 'Product')
                                    image_url = item.get('link')
                                    
                                    if not image_url:
                                        continue
                                    
                                    # Skip placeholder images
                                    if any(skip_term in image_url.lower() for skip_term in ['placeholder', 'no-image', 'default']):
                                        continue
                                    
                                    context = item.get('image', {})
                                    link = context.get('contextLink', image_url)
                                    snippet = item.get('snippet', title)
                                    
                                    price = extract_price_from_text(f"{title} {snippet}")
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
                            
                            print(f"  [OK] Found {len(products)} products from '{query}'")
                            break
                        
                    elif response.status_code == 429:
                        print(f"  [RATE LIMIT] Hit rate limit for current API key")
                        new_key = api_key_manager.rotate_key()
                        if new_key and new_key != api_key:
                            api_key = new_key
                            params["key"] = api_key
                            print(f"  [RATE LIMIT] Rotated to new API key, retrying...")
                            await rate_limiter.wait_if_needed(api_key)
                            continue
                        else:
                            wait_time = (2 ** attempt) + 1
                            print(f"  [RATE LIMIT] Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                    
                    else:
                        print(f"  [ERROR] API error: {response.status_code} for '{query}'")
                        break
                        
                except Exception as e:
                    print(f"  [ERROR] Request failed: {e} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        break
                    
        await search_cache.set(query, page, products)
        
    except Exception as e:
        print(f"  [ERROR] Search error: {e}")
    
    print(f"[SEARCH] Completed search for '{query}' page {page}: found {len(products)} products")
    return products

def extract_price_from_text(text: str) -> str:
    """Extract price from text"""
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
    noise_patterns = [
        r'\|\s*.*$',
        r'\-\s*.*$',
        r'\d{3,}x\d{3,}',
    ]
    
    for pattern in noise_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    return title.strip()[:150]

def detect_source_from_url(url: str) -> str:
    """Detect the e-commerce platform from URL"""
    if not url:
        return "Unknown"
    
    url_lower = url.lower()
    
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
    else:
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

def generate_search_queries(category_info: dict, colors: List[str], broad_category: str, gender: str) -> List[str]:
    """Generate targeted search queries based on detected product with gender and category"""
    primary_category = category_info.get("primary_category", "product")
    
    queries = []
    
    # Clean the primary category by removing gender prefixes for better search
    clean_category = primary_category
    if "women's" in clean_category.lower():
        clean_category = clean_category.replace("women's", "").replace("womens", "").strip()
    elif "men's" in clean_category.lower():
        clean_category = clean_category.replace("men's", "").replace("mens", "").strip()
    
    # Base queries with gender context
    if gender != "unisex":
        # Gender-specific queries
        if colors:
            for color in colors[:2]:
                if color not in ["multi-color", "mixed"]:
                    queries.append(f"{gender} {color} {clean_category}")
                    queries.append(f"{color} {clean_category} for {gender}")
        else:
            queries.append(f"{gender} {clean_category}")
        
        # Platform-specific with gender
        platforms = ["Amazon", "Flipkart", "Myntra", "Ajio", "Meesho"]
        for platform in platforms:
            queries.append(f"{gender} {clean_category} {platform}")
        
        # General gender-specific
        queries.append(f"{gender} fashion {clean_category}")
        queries.append(f"buy {gender} {clean_category} online")
    
    # Color-focused queries (without gender when ambiguous)
    if colors:
        for color in colors[:2]:
            if color not in ["multi-color", "mixed"]:
                queries.append(f"{color} {clean_category}")
                queries.append(f"{clean_category} {color}")
    
    # Fallback queries
    queries.extend([
        f"{clean_category}",
        f"buy {clean_category} online",
        f"{clean_category} shopping",
        f"{clean_category} 2024"
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for query in queries:
        if query.lower() not in seen and len(query) > 3:
            seen.add(query.lower())
            unique_queries.append(query)
    
    print(f"  [QUERY DEBUG] First 5 queries: {unique_queries[:5]}")
    return unique_queries[:15]

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
                
                response = await client.get(product["image_url"])
                if response.status_code == 200:
                    embedding = extract_clip_embedding(response.content)
                    
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
                if idx < 5:
                    print(f"  [WARNING] Error on product {idx+1}: {str(e)[:50]}")
                continue
    
    if points:
        print(f"  Uploading {len(points)} vectors to Qdrant...")
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

# ────────────────────────────────
# IMPROVED PRODUCT SEARCH ENDPOINT
# ────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search_similar_products(image: UploadFile = File(...)):
    """Upload an image and find visually similar products with improved filtering"""
    try:
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print("[OK] Database cleared")
        except Exception as e:
            print(f"[WARNING] Database clear error: {e}")
        
        image_bytes = await image.read()
        
        print("\n" + "="*70)
        print("[VISUAL PRODUCT SEARCH]")
        print("="*70)
        
        img = Image.open(io.BytesIO(image_bytes))
        
        print("\n[ANALYZE] Analyzing image with CLIP...")
        category_info = analyze_image_with_clip(img)
        colors = extract_dominant_colors(img)
        
        print(f"  Detected: {category_info['primary_category']}")
        print(f"  Confidence: {category_info['confidence']:.2%}")
        print(f"  Category: {category_info['broad_category']}")
        print(f"  Gender: {category_info['gender']}")
        print(f"  Colors: {', '.join(colors)}")
        
        print(f"\n[QUERY] Generating search queries...")
        search_queries = generate_search_queries(
            category_info, 
            colors, 
            category_info['broad_category'],
            category_info['gender']
        )
        print(f"  Generated {len(search_queries)} queries")
        
        print(f"\n[GOOGLE] Searching Google Custom Search API...")
        all_products = []
        max_products = 80
        
        for i, query in enumerate(search_queries):
            if len(all_products) >= max_products:
                break
            
            products = await fetch_from_google_custom_search(query, 1)
            all_products.extend(products)
            
            # Small delay between queries
            await asyncio.sleep(0.3)
        
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
        
        # Apply attribute filtering BEFORE indexing
        print(f"\n[FILTER] Applying attribute-based filtering...")
        filtered_products = filter_products_by_attributes(
            unique_products, 
            colors, 
            category_info['gender'],
            category_info['broad_category']
        )
        
        print(f"  After attribute filtering: {len(filtered_products)} products")
        
        query_embedding = extract_clip_embedding(image_bytes)
        
        # Use filtered products for indexing, fallback to all if filtering is too strict
        products_to_index = filtered_products if len(filtered_products) > 10 else unique_products
        indexed_count = await index_products(products_to_index)
        
        if indexed_count == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to index products"
            )
        
        print(f"\n[MATCH] Finding visually similar products...")
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=40
        )
        
        print(f"  Found {len(search_results)} candidates")
        
        # Dynamic similarity threshold based on category
        min_threshold = 0.55
        if category_info['broad_category'] in ['women_fashion', 'men_fashion']:
            min_threshold = 0.60
        elif category_info['broad_category'] == 'electronics':
            min_threshold = 0.58
        
        # Filter by similarity and remove duplicates
        filtered_results = []
        seen_titles = set()
        
        for hit in search_results:
            if len(filtered_results) >= 25:
                break
                
            if hit.score >= min_threshold:
                title_lower = hit.payload['title'].lower()
                if title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    filtered_results.append(hit)
        
        print(f"  After similarity filtering: {len(filtered_results)} products")
        print(f"  Similarity threshold: {min_threshold:.2f}")
        
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
        print(f"  Primary match: {category_info['primary_category']}")
        print(f"  Gender: {category_info['gender']}")
        print(f"  Colors: {colors}")
        print("="*70 + "\n")
        
        return SearchResponse(
            results=results[:20],  # Return top 20 results
            total_scraped=len(unique_products),
            sources=list(sources),
            detected_category=category_info['primary_category'],
            detected_attributes={
                "colors": colors,
                "gender": category_info['gender'],
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

# ... (keep the rest of your endpoints the same)

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
        existing = await collection_collection.find_one({
            "user_id": user_id,
            "image_url": image_url
        })
        
        if existing:
            return {"message": "Image already in collection", "collection_id": str(existing["_id"])}
        
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

# Virtual Try-On endpoints
@app.post("/api/tryon")
async def try_on(
    human_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    """Virtual try-on endpoint"""
    temp_human_path = None
    temp_garm_path = None
    
    try:
        print("\n[TRYON] Starting virtual try-on process...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_human:
            temp_human.write(await human_image.read())
            temp_human_path = temp_human.name
            print(f"  Saved human image: {temp_human_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_garm:
            temp_garm.write(await garment_image.read())
            temp_garm_path = temp_garm.name
            print(f"  Saved garment image: {temp_garm_path}")

        print("  Processing with Leffa model...")
        output_path = process_tryon_image(temp_human_path, temp_garm_path)
        
        print(f"  [OK] Try-on complete: {output_path}")
        
        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        print(f"  [ERROR] Try-on failed: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if temp_human_path and os.path.exists(temp_human_path):
            try:
                os.unlink(temp_human_path)
            except:
                pass
        if temp_garm_path and os.path.exists(temp_garm_path):
            try:
                os.unlink(temp_garm_path)
            except:
                pass


@app.post("/api/tryon/from-collection")
async def try_on_from_collection(
    human_image: UploadFile = File(...),
    garment_url: str = Form(...),
    user_id: str = Form("anonymous")
):
    """
    Try on a garment from a URL (e.g., user's collection).
    - human_image: uploaded portrait/full-body image
    - garment_url: image URL of the garment to try on
    - user_id: optional identifier (unused here but reserved)
    """
    temp_human_path = None
    temp_garm_path = None
    try:
        if not garment_url:
            raise HTTPException(status_code=400, detail="garment_url is required")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_human:
            temp_human.write(await human_image.read())
            temp_human_path = temp_human.name

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(garment_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download garment image")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_garm:
                temp_garm.write(resp.content)
                temp_garm_path = temp_garm.name

        output_path = process_tryon_image(temp_human_path, temp_garm_path)
        return FileResponse(output_path, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if temp_human_path and os.path.exists(temp_human_path):
            try:
                os.unlink(temp_human_path)
            except:
                pass
        if temp_garm_path and os.path.exists(temp_garm_path):
            try:
                os.unlink(temp_garm_path)
            except:
                pass

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

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Visual Product Search & Virtual Try-On API 🛍️👗",
        "version": "1.0",
        "endpoints": {
            "product_search": "/search",
            "virtual_tryon": "/api/tryon",
            "collection": {
                "add": "/collection/add",
                "list": "/collection/list",
                "remove": "/collection/remove/{id}",
                "clear": "/collection/clear"
            },
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")