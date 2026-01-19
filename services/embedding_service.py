"""
Azure OpenAI Embedding Service
Generates embeddings using Azure OpenAI text-embedding-ada-002 model.
"""
import logging
from typing import List, Optional, Union
import numpy as np
from openai import AzureOpenAI
from config.settings import CONFIG

logger = logging.getLogger("document_ai")


class EmbeddingService:
    """
    Service for generating embeddings using Azure OpenAI.
    Uses text-embedding-ada-002 which produces 1536-dimensional vectors.
    """
    
    EMBEDDING_DIMENSION = 1536  # ada-002 output dimension
    
    def __init__(self):
        self.client: Optional[AzureOpenAI] = None
        self.deployment_name: str = CONFIG.get("azure_embedding_deployment_name", "text-embedding-ada-002")
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client for embeddings."""
        try:
            endpoint = CONFIG.get("azure_openai_endpoint")
            api_key = CONFIG.get("azure_openai_api_key")
            api_version = CONFIG.get("azure_embedding_api_version", "2023-05-15")
            
            if not endpoint or not api_key:
                logger.error("❌ Azure OpenAI credentials not configured")
                raise ValueError("Missing Azure OpenAI credentials")
            
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            
            logger.info(f"✅ Embedding service initialized (deployment: {self.deployment_name})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding service: {e}")
            raise
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector, or None on failure
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for embedding")
            return None
        
        try:
            # Truncate text if too long (ada-002 has 8191 token limit)
            # Approximate: 1 token ≈ 4 characters
            max_chars = 8000 * 4
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.warning(f"⚠️ Text truncated to {max_chars} characters for embedding")
            
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (max 16 for Azure)
            
        Returns:
            List of embedding vectors (None for failed texts)
        """
        if not texts:
            return []
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Clean and truncate texts
                cleaned_batch = []
                for text in batch:
                    if text and text.strip():
                        # Truncate if needed
                        max_chars = 8000 * 4
                        cleaned_batch.append(text[:max_chars] if len(text) > max_chars else text)
                    else:
                        cleaned_batch.append("")
                
                # Skip empty batches
                non_empty_indices = [j for j, t in enumerate(cleaned_batch) if t]
                if not non_empty_indices:
                    embeddings.extend([None] * len(batch))
                    continue
                
                non_empty_texts = [cleaned_batch[j] for j in non_empty_indices]
                
                response = self.client.embeddings.create(
                    input=non_empty_texts,
                    model=self.deployment_name
                )
                
                # Map results back to original positions
                batch_embeddings = [None] * len(batch)
                for idx, emb_data in enumerate(response.data):
                    original_idx = non_empty_indices[idx]
                    batch_embeddings[original_idx] = emb_data.embedding
                
                embeddings.extend(batch_embeddings)
                logger.debug(f"📊 Embedded batch {batch_num}/{total_batches} ({len(non_empty_texts)} texts)")
                
            except Exception as e:
                logger.error(f"❌ Batch embedding failed (batch {batch_num}): {e}")
                embeddings.extend([None] * len(batch))
        
        logger.info(f"✅ Generated {sum(1 for e in embeddings if e is not None)}/{len(texts)} embeddings")
        return embeddings
    
    def compute_cosine_similarity(
        self, 
        embedding1: Union[List[float], np.ndarray], 
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity computation failed: {e}")
            return 0.0


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
