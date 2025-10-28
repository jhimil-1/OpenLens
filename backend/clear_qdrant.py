#!/usr/bin/env python3
"""
Clear Qdrant collection script
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_qdrant_collection():
    """Clear the product embeddings collection"""
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        collection_name = "product_embeddings"
        
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        print(f"Current collections: {collection_names}")
        
        if collection_name in collection_names:
            print(f"🗑️ Deleting collection: {collection_name}")
            qdrant_client.delete_collection(collection_name)
            print(f"✅ Collection {collection_name} deleted successfully")
        else:
            print(f"ℹ️ Collection {collection_name} does not exist")
            
        # Verify deletion
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        print(f"Remaining collections: {collection_names}")
        
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")

if __name__ == "__main__":
    clear_qdrant_collection()