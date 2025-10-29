import React, { useState, useEffect } from 'react';
import { Upload, Search, AlertCircle, Loader2, ExternalLink, DollarSign, ShoppingCart, X, Trash2 } from 'lucide-react';

export default function VisualProductSearch() {
  const styles = `
    .error-message {
      background-color: #fee;
      color: #c33;
      padding: 1rem;
      border-radius: 8px;
      margin: 1rem 0;
    }
    .results-section {
      margin-top: 2rem;
    }
    .detection-info {
      background-color: #f8f9ff;
      border: 1px solid #e0e6ff;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1.5rem;
      color: #4a5568;
    }
    .detection-info p {
      margin: 0.5rem 0;
    }
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-top: 1rem;
    }
  `;
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [searchStatus, setSearchStatus] = useState('');
  const [detectedInfo, setDetectedInfo] = useState(null);
  const [collection, setCollection] = useState([]);
  const [addedToCollection, setAddedToCollection] = useState({});
  const [showCollection, setShowCollection] = useState(false);

  // Load collection from backend on component mount
  useEffect(() => {
    loadCollectionFromBackend();
  }, []);

  const loadCollectionFromBackend = async () => {
    try {
      const response = await fetch('http://localhost:8003/collection/list');
      if (response.ok) {
        const data = await response.json();
        setCollection(data.items.map(item => ({
          image_url: item.image_url,
          collectionId: item.collection_id
        })));
      }
    } catch (error) {
      console.error('Error loading collection from backend:', error);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResults([]);
      setError(null);
      setDetectedInfo(null);
    }
  };

  const addToCollection = async (product) => {
    try {
      const response = await fetch(`http://localhost:8003/collection/add?image_url=${encodeURIComponent(product.image_url)}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const data = await response.json();
        setCollection(prevCollection => [...prevCollection, {
          image_url: product.image_url,
          collectionId: data.collection_id
        }]);
        setAddedToCollection(prev => ({ ...prev, [product.image_url]: true }));
        
        // Remove "added" status after 2 seconds
        setTimeout(() => {
          setAddedToCollection(prev => {
            const newState = { ...prev };
            delete newState[product.image_url];
            return newState;
          });
        }, 2000);
      }
    } catch (error) {
      console.error('Error adding to collection:', error);
    }
  };

  const removeFromCollection = async (collectionId) => {
    try {
      const response = await fetch(`http://localhost:8003/collection/remove/${collectionId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        setCollection(prevCollection => prevCollection.filter(item => item.collectionId !== collectionId));
      }
    } catch (error) {
      console.error('Error removing from collection:', error);
    }
  };

  const handleSearch = async () => {
    if (!selectedImage) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setSearchStatus('Extracting CLIP embeddings...');

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch('http://localhost:8003/search', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results);
      setSearchStatus(`Found ${data.total_scraped} products from ${data.sources.join(', ')}`);
      setDetectedInfo({
        category: data.detected_category,
        colors: data.detected_attributes?.colors || [],
        attributes: data.detected_attributes
      });
    } catch (err) {
      setError(err.message);
      setSearchStatus('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-4xl font-bold text-gray-900">
              Visual Product Search
            </h1>
            <button
              onClick={() => setShowCollection(true)}
              className={`px-4 py-2 rounded-full flex items-center gap-2 transition-colors ${
                collection.length > 0
                  ? 'bg-green-600 hover:bg-green-700 text-white cursor-pointer'
                  : 'bg-gray-200 text-gray-500 cursor-not-allowed'
              }`}
              disabled={collection.length === 0}
            >
              <ShoppingCart size={20} />
              <span className="font-semibold">{collection.length} items</span>
            </button>
          </div>
          <p className="text-gray-600">
            Upload an image to find visually similar products using AI-powered CLIP embeddings
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Product Image
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-purple-500 transition-colors">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  id="image-upload"
                />
                <label htmlFor="image-upload" className="cursor-pointer">
                  {imagePreview ? (
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="max-h-64 mx-auto rounded-lg"
                    />
                  ) : (
                    <div>
                      <Upload className="mx-auto h-12 w-12 text-gray-400 mb-2" />
                      <p className="text-gray-600">Click to upload an image</p>
                      <p className="text-sm text-gray-500 mt-1">PNG, JPG up to 10MB</p>
                    </div>
                  )}
                </label>
              </div>
            </div>

            <div className="flex-1 flex flex-col justify-center">
              <button
                onClick={handleSearch}
                disabled={!selectedImage || loading}
                className="bg-purple-600 text-white px-8 py-4 rounded-lg font-semibold hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search size={20} />
                    Find Similar Products
                  </>
                )}
              </button>

              {searchStatus && (
                <p className="text-sm text-purple-600 mt-4 text-center animate-pulse">
                  {searchStatus}
                </p>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-50 rounded-lg flex items-start gap-2">
                  <AlertCircle className="text-red-600 flex-shrink-0" size={20} />
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {results.length > 0 && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              🔍 Found {results.length} Similar Products
            </h2>
            {detectedInfo?.category && (
              <div className="mb-4 p-3 bg-purple-50 rounded-lg border border-purple-200">
                <p className="text-purple-800 font-medium">
                  📦 Detected Category: <strong>{detectedInfo.category.replace('_', ' ').toUpperCase()}</strong>
                </p>
                {detectedInfo?.colors && detectedInfo.colors.length > 0 && (
                  <p className="text-purple-700 text-sm mt-1">
                    🎨 Detected Colors: <strong>{detectedInfo.colors.join(', ').toUpperCase()}</strong>
                  </p>
                )}
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {results.map((product, index) => (
                <div
                  key={index}
                  className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
                >
                  <div className="aspect-square bg-gray-100 overflow-hidden">
                    <img
                      src={product.image_url}
                      alt={product.title}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect width="400" height="400" fill="%23ddd"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999" font-size="18"%3ENo Image%3C/text%3E%3C/svg%3E';
                      }}
                    />
                  </div>
                  <div className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-semibold text-gray-900 line-clamp-2 flex-1">
                        {product.title}
                      </h3>
                      <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-800 text-xs font-medium rounded">
                        {(product.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">
                        {product.source}
                      </span>
                      {product.price && (
                        <div className="flex items-center gap-1 text-green-600 font-bold text-sm">
                          <DollarSign size={14} />
                          <span>{product.price}</span>
                        </div>
                      )}
                    </div>
                    
                    {product.description && (
                      <p className="text-gray-600 text-sm line-clamp-3 mb-3">
                        {product.description}
                      </p>
                    )}
                    
                    <div className="flex gap-2">
                      <a
                        href={product.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex-1 inline-flex items-center justify-center gap-2 text-purple-600 hover:text-purple-800 font-medium text-sm border border-purple-200 rounded-lg px-3 py-2 hover:bg-purple-50 transition-colors"
                      >
                        View Product
                        <ExternalLink size={16} />
                      </a>
                      <button
                        onClick={() => addToCollection(product)}
                        disabled={addedToCollection[product.image_url]}
                        className={`inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg font-medium text-sm transition-colors ${
                          addedToCollection[product.image_url]
                            ? 'bg-green-500 text-white cursor-not-allowed'
                            : 'bg-orange-500 hover:bg-orange-600 text-white'
                        }`}
                      >
                        {addedToCollection[product.image_url] ? (
                          <>
                            <ExternalLink size={16} />
                            Added!
                          </>
                        ) : (
                          <>
                            <ShoppingCart size={16} />
                            Add to Collection
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Collection Modal */}
      {showCollection && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Image Collection</h2>
              <button
                onClick={() => setShowCollection(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            {collection.length === 0 ? (
              <p className="text-gray-500 text-center py-8">Your collection is empty</p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {collection.map((item) => (
                  <div key={item.collectionId} className="relative group">
                    <img
                      src={item.image_url}
                      alt="Collection item"
                      className="w-full h-40 object-cover rounded-lg"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-opacity rounded-lg flex items-center justify-center">
                      <button
                        onClick={() => removeFromCollection(item.collectionId)}
                        className="opacity-0 group-hover:opacity-100 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-opacity"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}