import React, { useState, useEffect } from 'react';
import { Upload, Search, AlertCircle, Loader2, ExternalLink, DollarSign, ShoppingCart, X, Trash2, User, Sparkles, Tag, Palette, Package } from 'lucide-react';

export default function VisualProductSearch() {
  const API_BASE_URL = 'http://localhost:8003';
  const USER_ID = 'anonymous';

  // Product search states
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [searchStatus, setSearchStatus] = useState('');
  const [detectedInfo, setDetectedInfo] = useState(null);
  const [totalScraped, setTotalScraped] = useState(0);
  const [sources, setSources] = useState([]);
  
  // Collection states
  const [collection, setCollection] = useState([]);
  const [addedToCollection, setAddedToCollection] = useState({});
  const [showCollection, setShowCollection] = useState(false);
  const [collectionLoading, setCollectionLoading] = useState(false);

  // Virtual Try-On states
  const [showTryOn, setShowTryOn] = useState(false);
  const [humanImage, setHumanImage] = useState(null);
  const [humanImagePreview, setHumanImagePreview] = useState(null);
  const [selectedGarment, setSelectedGarment] = useState(null);
  const [tryOnLoading, setTryOnLoading] = useState(false);
  const [tryOnResult, setTryOnResult] = useState(null);
  const [tryOnError, setTryOnError] = useState(null);

  // Load collection on mount
  useEffect(() => {
    loadCollectionFromBackend();
  }, []);

  // Collection API functions
  const loadCollectionFromBackend = async () => {
    setCollectionLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/collection/list?user_id=${USER_ID}`);
      if (response.ok) {
        const data = await response.json();
        setCollection(data.items || []);
      }
    } catch (error) {
      console.error('Error loading collection:', error);
    } finally {
      setCollectionLoading(false);
    }
  };

  const addToCollection = async (product) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/collection/add?image_url=${encodeURIComponent(product.image_url)}&user_id=${USER_ID}`,
        { method: 'POST' }
      );
      
      if (response.ok) {
        const data = await response.json();
        setCollection(prev => [...prev, {
          image_url: product.image_url,
          collection_id: data.collection_id,
          created_at: new Date().toISOString()
        }]);
        setAddedToCollection(prev => ({ ...prev, [product.image_url]: true }));
        
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
      const response = await fetch(
        `${API_BASE_URL}/collection/remove/${collectionId}?user_id=${USER_ID}`,
        { method: 'DELETE' }
      );
      
      if (response.ok) {
        setCollection(prev => prev.filter(item => item.collection_id !== collectionId));
      }
    } catch (error) {
      console.error('Error removing from collection:', error);
    }
  };

  const clearCollection = async () => {
    if (!window.confirm('Are you sure you want to clear your entire collection?')) {
      return;
    }
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/collection/clear?user_id=${USER_ID}`,
        { method: 'DELETE' }
      );
      
      if (response.ok) {
        setCollection([]);
      }
    } catch (error) {
      console.error('Error clearing collection:', error);
    }
  };

  // Image upload handlers
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
      setSearchStatus('');
    }
  };

  const handleHumanImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setHumanImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setHumanImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      setTryOnResult(null);
      setTryOnError(null);
    }
  };

  // Product search
  const handleSearch = async () => {
    if (!selectedImage) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setSearchStatus('Analyzing image with CLIP...');

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      setResults(data.results || []);
      setTotalScraped(data.total_scraped || 0);
      setSources(data.sources || []);
      setDetectedInfo({
        category: data.detected_category,
        colors: data.detected_attributes?.colors || [],
        broad_category: data.detected_attributes?.broad_category,
        confidence: data.detected_attributes?.confidence
      });
      
      setSearchStatus(`✓ Found ${data.results?.length || 0} similar products from ${data.total_scraped || 0} scraped`);
    } catch (err) {
      setError(err.message);
      setSearchStatus('');
    } finally {
      setLoading(false);
    }
  };

  // Virtual Try-On
  const handleTryOn = async () => {
    if (!humanImage || !selectedGarment) {
      setTryOnError('Please upload your photo and select a garment');
      return;
    }

    setTryOnLoading(true);
    setTryOnError(null);
    setTryOnResult(null);

    try {
      const formData = new FormData();
      formData.append('human_image', humanImage);
      formData.append('garment_url', selectedGarment);

      const response = await fetch('http://localhost:8003/api/tryon/from-collection', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Try-on failed: ${response.statusText}`);
      }

      const resultBlob = await response.blob();
      const resultUrl = URL.createObjectURL(resultBlob);
      setTryOnResult(resultUrl);
    } catch (err) {
      setTryOnError(err.message);
    } finally {
      setTryOnLoading(false);
    }
  };

  // Try-On from collection
  const handleTryOnFromCollection = async () => {
    if (!humanImage || !selectedGarment) {
      setTryOnError('Please upload your photo and select a garment');
      return;
    }

    setTryOnLoading(true);
    setTryOnError(null);
    setTryOnResult(null);

    try {
      const formData = new FormData();
      formData.append('human_image', humanImage);
      formData.append('garment_url', selectedGarment);
      formData.append('user_id', USER_ID);

      const response = await fetch(`${API_BASE_URL}/api/tryon/from-collection`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Try-on failed: ${response.statusText}`);
      }

      const resultBlob = await response.blob();
      const resultUrl = URL.createObjectURL(resultBlob);
      setTryOnResult(resultUrl);
    } catch (err) {
      setTryOnError(err.message);
    } finally {
      setTryOnLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex flex-col md:flex-row justify-between items-center mb-4 gap-4">
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 flex items-center gap-2">
              <Search className="text-purple-600" />
              Visual Product Search
            </h1>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowTryOn(true);
                  loadCollectionFromBackend();
                }}
                className={`px-4 py-2 rounded-full flex items-center gap-2 transition-all transform hover:scale-105 ${
                  collection.length > 0
                    ? 'bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-lg'
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                }`}
                disabled={collection.length === 0}
              >
                <User size={20} />
                <span className="font-semibold">Virtual Try-On</span>
              </button>
              <button
                onClick={() => {
                  setShowCollection(true);
                  loadCollectionFromBackend();
                }}
                className={`px-4 py-2 rounded-full flex items-center gap-2 transition-all transform hover:scale-105 ${
                  collection.length > 0
                    ? 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white shadow-lg'
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                }`}
                disabled={collection.length === 0}
              >
                <ShoppingCart size={20} />
                <span className="font-semibold">{collection.length} items</span>
              </button>
            </div>
          </div>
          <p className="text-gray-600 text-sm md:text-base">
            🚀 Upload an image to find visually similar products using AI-powered CLIP embeddings
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-xl p-6 md:p-8 mb-8">
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Product Image
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-purple-500 transition-colors cursor-pointer">
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
                      className="max-h-64 mx-auto rounded-lg shadow-md"
                    />
                  ) : (
                    <div>
                      <Upload className="mx-auto h-12 w-12 text-gray-400 mb-2" />
                      <p className="text-gray-600 font-medium">Click to upload an image</p>
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
                className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-8 py-4 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center gap-2 shadow-lg"
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
                <div className="mt-4 p-3 bg-purple-50 rounded-lg border border-purple-200">
                  <p className="text-sm text-purple-700 text-center font-medium">
                    {searchStatus}
                  </p>
                </div>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-50 rounded-lg flex items-start gap-2 border border-red-200">
                  <AlertCircle className="text-red-600 flex-shrink-0" size={20} />
                  <div>
                    <p className="text-red-800 text-sm font-semibold">Error</p>
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Detection Info */}
        {detectedInfo && (
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl p-6 mb-6 border border-purple-200 shadow-md">
            <h3 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
              <Package className="text-purple-600" />
              AI Detection Results
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-2">
                <Tag className="text-blue-600" size={20} />
                <div>
                  <p className="text-xs text-gray-600">Category</p>
                  <p className="font-semibold text-gray-900">
                    {detectedInfo.category?.replace('_', ' ').toUpperCase()}
                  </p>
                </div>
              </div>
              {detectedInfo.colors && detectedInfo.colors.length > 0 && (
                <div className="flex items-center gap-2">
                  <Palette className="text-pink-600" size={20} />
                  <div>
                    <p className="text-xs text-gray-600">Colors</p>
                    <p className="font-semibold text-gray-900">
                      {detectedInfo.colors.join(', ').toUpperCase()}
                    </p>
                  </div>
                </div>
              )}
              {detectedInfo.confidence && (
                <div className="flex items-center gap-2">
                  <Sparkles className="text-yellow-600" size={20} />
                  <div>
                    <p className="text-xs text-gray-600">Confidence</p>
                    <p className="font-semibold text-gray-900">
                      {(detectedInfo.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                🔍 Found {results.length} Similar Products
              </h2>
              {sources.length > 0 && (
                <div className="hidden md:flex items-center gap-2 text-sm text-gray-600">
                  <span className="font-medium">Sources:</span>
                  {sources.slice(0, 3).map((source, idx) => (
                    <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                      {source}
                    </span>
                  ))}
                  {sources.length > 3 && (
                    <span className="text-xs text-gray-500">+{sources.length - 3} more</span>
                  )}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {results.map((product, index) => (
                <div
                  key={index}
                  className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all transform hover:-translate-y-1"
                >
                  <div className="aspect-square bg-gray-100 overflow-hidden relative">
                    <img
                      src={product.image_url}
                      alt={product.title}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect width="400" height="400" fill="%23ddd"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999" font-size="18"%3ENo Image%3C/text%3E%3C/svg%3E';
                      }}
                    />
                    <div className="absolute top-2 right-2 px-3 py-1 bg-purple-600 text-white text-xs font-bold rounded-full shadow-lg">
                      {(product.similarity * 100).toFixed(1)}% Match
                    </div>
                  </div>
                  <div className="p-4">
                    <h3 className="font-semibold text-gray-900 line-clamp-2 mb-2 min-h-[3rem]">
                      {product.title}
                    </h3>
                    
                    <div className="flex items-center gap-2 mb-3 flex-wrap">
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
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
                      <p className="text-gray-600 text-sm line-clamp-2 mb-3">
                        {product.description}
                      </p>
                    )}
                    
                    <div className="flex gap-2">
                      <a
                        href={product.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex-1 inline-flex items-center justify-center gap-2 text-purple-600 hover:text-purple-800 font-medium text-sm border-2 border-purple-200 rounded-lg px-3 py-2 hover:bg-purple-50 transition-all"
                      >
                        View Product
                        <ExternalLink size={16} />
                      </a>
                      <button
                        onClick={() => addToCollection(product)}
                        disabled={addedToCollection[product.image_url]}
                        className={`inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg font-medium text-sm transition-all ${
                          addedToCollection[product.image_url]
                            ? 'bg-green-500 text-white cursor-not-allowed'
                            : 'bg-orange-500 hover:bg-orange-600 text-white transform hover:scale-105'
                        }`}
                      >
                        {addedToCollection[product.image_url] ? (
                          <>
                            ✓ Added
                          </>
                        ) : (
                          <>
                            <ShoppingCart size={16} />
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
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold flex items-center gap-2">
                <ShoppingCart className="text-green-600" />
                My Collection ({collection.length})
              </h2>
              <div className="flex gap-2">
                {collection.length > 0 && (
                  <button
                    onClick={clearCollection}
                    className="px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                  >
                    <Trash2 size={16} />
                    Clear All
                  </button>
                )}
                <button
                  onClick={() => setShowCollection(false)}
                  className="text-gray-500 hover:text-gray-700 p-2"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>
            
            {collectionLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="animate-spin text-purple-600" size={32} />
              </div>
            ) : collection.length === 0 ? (
              <div className="text-center py-12">
                <ShoppingCart className="mx-auto text-gray-300 mb-4" size={48} />
                <p className="text-gray-500 text-lg">Your collection is empty</p>
                <p className="text-gray-400 text-sm mt-2">Add products from search results to see them here</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {collection.map((item) => (
                  <div key={item.collection_id} className="relative group">
                    <img
                      src={item.image_url}
                      alt="Collection item"
                      className="w-full h-40 object-cover rounded-lg shadow-md"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-opacity rounded-lg flex items-center justify-center">
                      <button
                        onClick={() => removeFromCollection(item.collection_id)}
                        className="opacity-0 group-hover:opacity-100 bg-red-500 text-white p-3 rounded-full hover:bg-red-600 transition-all transform hover:scale-110 shadow-lg"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Virtual Try-On Modal */}
      {showTryOn && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-xl p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl md:text-3xl font-bold flex items-center gap-2">
                <Sparkles className="text-blue-600" />
                Virtual Try-On
              </h2>
              <button
                onClick={() => {
                  setShowTryOn(false);
                  setHumanImage(null);
                  setHumanImagePreview(null);
                  setSelectedGarment(null);
                  setTryOnResult(null);
                  setTryOnError(null);
                }}
                className="text-gray-500 hover:text-gray-700 p-2"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* Upload Human Photo */}
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <User className="text-blue-600" size={20} />
                  1. Upload Your Photo
                </h3>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors cursor-pointer">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleHumanImageUpload}
                    className="hidden"
                    id="human-image-upload"
                  />
                  <label htmlFor="human-image-upload" className="cursor-pointer">
                    {humanImagePreview ? (
                      <img
                        src={humanImagePreview}
                        alt="Your photo"
                        className="max-h-64 mx-auto rounded-lg shadow-md"
                      />
                    ) : (
                      <div>
                        <User className="mx-auto h-12 w-12 text-gray-400 mb-2" />
                        <p className="text-gray-600 font-medium">Click to upload your photo</p>
                        <p className="text-sm text-gray-500 mt-1">Full body photo works best</p>
                      </div>
                    )}
                  </label>
                </div>
              </div>

              {/* Select Garment */}
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <ShoppingCart className="text-green-600" size={20} />
                  2. Select Garment from Collection
                </h3>
                <div className="border-2 border-gray-300 rounded-lg p-4 max-h-80 overflow-y-auto">
                  {collection.length === 0 ? (
                    <div className="text-center py-8">
                      <ShoppingCart className="mx-auto text-gray-300 mb-3" size={40} />
                      <p className="text-gray-500">No items in collection</p>
                      <p className="text-gray-400 text-sm mt-1">Add products first</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-3">
                      {collection.map((item) => (
                        <div
                          key={item.collection_id}
                          onClick={() => setSelectedGarment(item.image_url)}
                          className={`cursor-pointer rounded-lg overflow-hidden border-2 transition-all transform hover:scale-105 ${
                            selectedGarment === item.image_url
                              ? 'border-blue-500 ring-2 ring-blue-300 shadow-lg'
                              : 'border-transparent hover:border-gray-300'
                          }`}
                        >
                          <img
                            src={item.image_url}
                            alt="Garment"
                            className="w-full h-24 object-cover"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Try On Button */}
            <div className="mb-6">
              <button
                onClick={handleTryOnFromCollection}
                disabled={!humanImage || !selectedGarment || tryOnLoading}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center gap-2 shadow-lg"
              >
                {tryOnLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Generating Virtual Try-On... (This may take 30-60 seconds)
                  </>
                ) : (
                  <>
                    <Sparkles size={20} />
                    Generate Virtual Try-On
                  </>
                )}
              </button>
            </div>

            {/* Error Message */}
            {tryOnError && (
              <div className="bg-red-50 border-2 border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4 flex items-start gap-2">
                <AlertCircle className="flex-shrink-0 mt-0.5" size={20} />
                <div>
                  <p className="font-semibold">Error:</p>
                  <p className="text-sm">{tryOnError}</p>
                </div>
              </div>
            )}

            {/* Result */}
            {tryOnResult && (
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Sparkles className="text-yellow-500" />
                  ✨ Your Virtual Try-On Result
                </h3>
                <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-6 border-2 border-purple-200">
                  <img
                    src={tryOnResult}
                    alt="Try-on result"
                    className="max-w-full mx-auto rounded-lg shadow-2xl"
                  />
                  <div className="mt-6 text-center">
                    <a
                      href={tryOnResult}
                      download="virtual-tryon-result.png"
                      className="inline-flex items-center gap-2 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white px-6 py-3 rounded-lg font-semibold transition-all transform hover:scale-105 shadow-lg"
                    >
                      <Upload size={18} />
                      Download Result
                    </a>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}