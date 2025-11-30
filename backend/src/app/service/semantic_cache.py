"""
Semantic Cache for AI Responses

Uses embeddings to find similar cached requests and return cached responses
when similarity exceeds a threshold, reducing API calls and response times.
"""

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "semantic_cache.json"
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"
SIMILARITY_THRESHOLD = 0.92  # Minimum similarity to consider a cache hit
MAX_CACHE_SIZE = 500  # Maximum number of cached entries
CACHE_TTL = 86400 * 7  # Cache TTL in seconds (7 days)


@dataclass
class CacheEntry:
    """A single cache entry"""
    request: str
    response: str
    mode: str
    timestamp: float
    embedding_index: int
    hit_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CacheEntry':
        return cls(**data)


class SemanticCache:
    """
    Semantic cache that uses embeddings to find similar requests.
    
    Features:
    - Uses OpenAI embeddings for semantic similarity
    - Persists cache to disk
    - Automatic TTL-based expiration
    - LRU-style eviction when cache is full
    """
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.client = OpenAI()
        self.entries: List[CacheEntry] = []
        self.embeddings: Optional[np.ndarray] = None
        self._ensure_cache_dir()
        self._load_cache()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.entries = [CacheEntry.from_dict(e) for e in data]
                    print(f"[SemanticCache] Loaded {len(self.entries)} cached entries")
            
            if EMBEDDINGS_FILE.exists() and self.entries:
                self.embeddings = np.load(EMBEDDINGS_FILE)
                print(f"[SemanticCache] Loaded embeddings: {self.embeddings.shape}")
            else:
                self.embeddings = None
                
            # Clean expired entries on load
            self._clean_expired()
                
        except Exception as e:
            print(f"[SemanticCache] Error loading cache: {e}")
            self.entries = []
            self.embeddings = None
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump([e.to_dict() for e in self.entries], f)
            
            if self.embeddings is not None:
                np.save(EMBEDDINGS_FILE, self.embeddings)
                
        except Exception as e:
            print(f"[SemanticCache] Error saving cache: {e}")
    
    def _clean_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        valid_indices = []
        valid_entries = []
        
        for i, entry in enumerate(self.entries):
            if current_time - entry.timestamp < CACHE_TTL:
                entry.embedding_index = len(valid_entries)
                valid_entries.append(entry)
                valid_indices.append(i)
        
        if len(valid_entries) < len(self.entries):
            removed = len(self.entries) - len(valid_entries)
            print(f"[SemanticCache] Removed {removed} expired entries")
            self.entries = valid_entries
            
            if self.embeddings is not None and valid_indices:
                self.embeddings = self.embeddings[valid_indices]
            elif not valid_indices:
                self.embeddings = None
            
            self._save_cache()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"[SemanticCache] Error getting embedding: {e}")
            raise
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _find_similar(self, query_embedding: np.ndarray) -> Optional[Tuple[CacheEntry, float]]:
        """Find the most similar cached entry"""
        if self.embeddings is None or len(self.entries) == 0:
            return None
        
        # Calculate similarities with all cached embeddings
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            return self.entries[best_idx], float(best_similarity)
        
        return None
    
    def _evict_if_needed(self):
        """Evict least recently used entries if cache is full"""
        if len(self.entries) >= MAX_CACHE_SIZE:
            # Sort by hit_count (ascending) then timestamp (ascending)
            # Remove the least used and oldest entries
            sorted_entries = sorted(
                enumerate(self.entries),
                key=lambda x: (x[1].hit_count, x[1].timestamp)
            )
            
            # Remove bottom 10%
            remove_count = max(1, len(self.entries) // 10)
            remove_indices = set(idx for idx, _ in sorted_entries[:remove_count])
            
            # Keep entries not in remove set
            new_entries = []
            keep_indices = []
            for i, entry in enumerate(self.entries):
                if i not in remove_indices:
                    entry.embedding_index = len(new_entries)
                    new_entries.append(entry)
                    keep_indices.append(i)
            
            self.entries = new_entries
            if self.embeddings is not None:
                self.embeddings = self.embeddings[keep_indices]
            
            print(f"[SemanticCache] Evicted {remove_count} entries, {len(self.entries)} remaining")
    
    def get(self, request: str, mode: str = "auto") -> Optional[Dict[str, Any]]:
        """
        Try to get a cached response for the request.
        
        Returns:
            Dict with 'response', 'similarity', 'cached_request' if found, None otherwise
        """
        try:
            query_embedding = self._get_embedding(request)
            result = self._find_similar(query_embedding)
            
            if result:
                entry, similarity = result
                # Update hit count
                entry.hit_count += 1
                self._save_cache()
                
                print(f"\n{'='*60}")
                print(f"[SemanticCache] 💾 CACHE HIT")
                print(f"{'='*60}")
                print(f"  Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
                print(f"  Hit count: {entry.hit_count}")
                print(f"  Original request: {entry.request[:80]}{'...' if len(entry.request) > 80 else ''}")
                print(f"  Current request:  {request[:80]}{'...' if len(request) > 80 else ''}")
                print(f"  Cached response:  {entry.response[:100]}{'...' if len(entry.response) > 100 else ''}")
                print(f"{'='*60}\n")
                
                return {
                    "response": entry.response,
                    "similarity": similarity,
                    "cached_request": entry.request,
                    "mode": entry.mode
                }
            
            print(f"[SemanticCache] Cache MISS for: {request[:50]}{'...' if len(request) > 50 else ''}")
            return None
            
        except Exception as e:
            print(f"[SemanticCache] Error in get: {e}")
            return None
    
    def set(self, request: str, response: str, mode: str = "auto"):
        """
        Cache a response for a request.
        
        Args:
            request: The user's request
            response: The AI's response
            mode: The generation mode used
        """
        try:
            # Check if very similar entry already exists
            query_embedding = self._get_embedding(request)
            existing = self._find_similar(query_embedding)
            
            if existing and existing[1] > 0.98:
                # Update existing entry instead of creating new
                entry, _ = existing
                entry.response = response
                entry.timestamp = time.time()
                entry.mode = mode
                self._save_cache()
                print(f"[SemanticCache] Updated existing entry")
                return
            
            # Evict if needed
            self._evict_if_needed()
            
            # Create new entry
            new_entry = CacheEntry(
                request=request,
                response=response,
                mode=mode,
                timestamp=time.time(),
                embedding_index=len(self.entries)
            )
            
            self.entries.append(new_entry)
            
            # Add embedding
            if self.embeddings is None:
                self.embeddings = query_embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, query_embedding])
            
            self._save_cache()
            print(f"[SemanticCache] Cached new entry (total: {len(self.entries)})")
            
        except Exception as e:
            print(f"[SemanticCache] Error in set: {e}")
    
    def clear(self):
        """Clear all cached entries"""
        self.entries = []
        self.embeddings = None
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        if EMBEDDINGS_FILE.exists():
            EMBEDDINGS_FILE.unlink()
        print("[SemanticCache] Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(e.hit_count for e in self.entries)
        return {
            "total_entries": len(self.entries),
            "total_hits": total_hits,
            "cache_size_mb": (
                CACHE_FILE.stat().st_size / 1024 / 1024 if CACHE_FILE.exists() else 0
            ) + (
                EMBEDDINGS_FILE.stat().st_size / 1024 / 1024 if EMBEDDINGS_FILE.exists() else 0
            ),
            "similarity_threshold": self.similarity_threshold,
            "max_cache_size": MAX_CACHE_SIZE,
            "ttl_days": CACHE_TTL / 86400
        }


# Global cache instance
semantic_cache = SemanticCache()
