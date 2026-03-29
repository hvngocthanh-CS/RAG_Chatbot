"""
Semantic Chunking Service.
Advanced chunking strategy based on semantic similarity instead of fixed token counts.

Technique: Embed sentences → calculate cosine similarity → detect semantic breaks
Reference: "Text Segmentation using Semantic Similarity" (ACL 2023)
"""
import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunkConfig:
    """Configuration for semantic chunking."""
    max_chunk_size: int = 800  # Maximum tokens per chunk (hard limit)
    min_chunk_size: int = 100  # Minimum tokens per chunk
    similarity_threshold: float = 0.5  # Below this = semantic break
    overlap_sentences: int = 2  # Number of sentences to overlap


class SemanticChunker:
    """
    Semantic-aware text chunker.
    
    Key Innovation:
    - Detects semantic boundaries using embedding similarity
    - More intelligent than fixed token counts
    - Creates chunks at natural topic transitions
    
    Algorithm:
    1. Split text into sentences
    2. Embed each sentence (using lightweight embeddings)
    3. Calculate cosine similarity between consecutive sentences
    4. When similarity < threshold OR max_tokens reached → new chunk
    5. Overlap sentences for context continuity
    
    Best for: Long-form documents with clear topic transitions
    Examples: Research papers, technical docs, legal documents
    """
    
    def __init__(
        self,
        embedding_service=None,
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.5,
        overlap_sentences: int = 2,
        encoding_name: str = "cl100k_base"
    ):
        self.embedding_service = embedding_service
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_sentences = overlap_sentences
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None
            logger.warning("Tiktoken not available, using character estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        Handles common abbreviations like Dr., Mr., etc.
        """
        # Replace common abbreviations to avoid incorrect splits
        text = re.sub(r'\b(Dr|Mr|Ms|Mrs|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.', r'\1<PERIOD>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences if s.strip()]
        
        return sentences
    
    async def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Embed sentences using the embedding service.
        Returns: numpy array of shape (num_sentences, embedding_dim)
        """
        if not self.embedding_service:
            logger.warning("No embedding service available, falling back to basic chunking")
            return None
        
        try:
            # Batch embed all sentences at once
            embeddings = await self.embedding_service.embed_documents(sentences)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error embedding sentences: {e}")
            return None
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Returns: similarity score [0, 1]
        """
        # Cosine similarity formula: dot(A, B) / (norm(A) * norm(B))
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        return float(similarity)
    
    def _detect_semantic_breaks(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Detect semantic breaks in the text.
        
        Returns: List of indices where semantic breaks occur
        """
        breaks = [0]  # Always start at index 0
        
        for i in range(1, len(sentences)):
            similarity = self._calculate_similarity(embeddings[i-1], embeddings[i])
            
            if similarity < self.similarity_threshold:
                breaks.append(i)
                logger.debug(f"Semantic break detected at sentence {i} (similarity: {similarity:.3f})")
        
        return breaks
    
    async def chunk_text_semantic(
        self,
        text_blocks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Main semantic chunking algorithm.
        
        Steps:
        1. Extract text from blocks
        2. Split into sentences
        3. Embed sentences
        4. Detect semantic breaks
        5. Create chunks at breaks (respecting token limits)
        """
        chunks = []
        
        for block in text_blocks:
            block_text = block.get("text", "")
            block_type = block.get("type", "paragraph")
            
            # Special handling for titles/headings
            if block_type in ["title", "heading"]:
                chunks.append(self._create_chunk(
                    block_text,
                    metadata,
                    block,
                    len(chunks)
                ))
                continue
            
            # Split into sentences
            sentences = self._split_into_sentences(block_text)
            
            if not sentences:
                continue
            
            # Embed sentences
            embeddings = await self._embed_sentences(sentences)
            
            if embeddings is None:
                # Fallback to simple sentence-based chunking
                block_chunks = self._fallback_chunking(sentences, metadata, block, len(chunks))
                chunks.extend(block_chunks)
                continue
            
            # Detect semantic breaks
            break_indices = self._detect_semantic_breaks(sentences, embeddings)
            
            # Create chunks at semantic breaks
            semantic_chunks = self._create_semantic_chunks(
                sentences, 
                break_indices, 
                metadata, 
                block,
                len(chunks)
            )
            chunks.extend(semantic_chunks)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _create_semantic_chunks(
        self,
        sentences: List[str],
        break_indices: List[int],
        metadata: Dict[str, Any],
        block: Dict[str, Any],
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Create chunks at detected semantic breaks.
        Respects max_chunk_size hard limit.
        """
        chunks = []
        
        for i in range(len(break_indices)):
            start = break_indices[i]
            end = break_indices[i + 1] if i + 1 < len(break_indices) else len(sentences)
            
            # Get sentences for this chunk
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            # Check token limit
            token_count = self.count_tokens(chunk_text)
            
            if token_count <= self.max_chunk_size:
                # Chunk is within limit
                chunks.append(self._create_chunk(
                    chunk_text,
                    metadata,
                    block,
                    start_idx + len(chunks)
                ))
            else:
                # Chunk too large, need to split further
                sub_chunks = self._split_large_semantic_chunk(
                    chunk_sentences,
                    metadata,
                    block,
                    start_idx + len(chunks)
                )
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_semantic_chunk(
        self,
        sentences: List[str],
        metadata: Dict[str, Any],
        block: Dict[str, Any],
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Split a semantically coherent but token-wise large chunk.
        Uses sliding window to respect max_chunk_size.
        """
        chunks = []
        current_sentences = []
        
        for sentence in sentences:
            current_sentences.append(sentence)
            current_text = " ".join(current_sentences)
            
            if self.count_tokens(current_text) > self.max_chunk_size:
                # Remove last sentence and create chunk
                current_sentences.pop()
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        metadata,
                        block,
                        start_idx + len(chunks)
                    ))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_sentences) - self.overlap_sentences)
                current_sentences = current_sentences[overlap_start:] + [sentence]
        
        # Don't forget last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_text,
                    metadata,
                    block,
                    start_idx + len(chunks)
                ))
        
        return chunks
    
    def _fallback_chunking(
        self,
        sentences: List[str],
        metadata: Dict[str, Any],
        block: Dict[str, Any],
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback to simple sentence-based chunking when embedding fails.
        """
        chunks = []
        current_sentences = []
        
        for sentence in sentences:
            current_sentences.append(sentence)
            current_text = " ".join(current_sentences)
            
            if self.count_tokens(current_text) >= self.max_chunk_size:
                current_sentences.pop()
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        metadata,
                        block,
                        start_idx + len(chunks)
                    ))
                current_sentences = [sentence]
        
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_text,
                    metadata,
                    block,
                    start_idx + len(chunks)
                ))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        block: Dict[str, Any],
        chunk_index: int
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            "content": text.strip(),
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_type": block.get("type", "text"),
                "page_number": block.get("page_number"),
                "section": block.get("section", ""),
                "token_count": self.count_tokens(text),
                "chunking_method": "semantic"
            }
        }


# ============================================================================
# COMPARISON: Token-based vs Semantic Chunking
# ============================================================================

class ChunkingComparison:
    """
    Helper class to compare different chunking strategies.
    
    Usage for interviews/learning:
    ```python
    comparison = ChunkingComparison()
    results = await comparison.compare_strategies(document_text)
    print(results)
    ```
    """
    
    @staticmethod
    async def compare_strategies(
        text_blocks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        embedding_service=None
    ) -> Dict[str, Any]:
        """
        Compare token-based vs semantic chunking.
        Returns metrics for both strategies.
        """
        from backend.services.chunker import TextChunker
        
        # Token-based chunking
        token_chunker = TextChunker(chunk_size=600, chunk_overlap=100)
        token_chunks = token_chunker.chunk_text(text_blocks, metadata)
        
        # Semantic chunking
        semantic_chunker = SemanticChunker(
            embedding_service=embedding_service,
            max_chunk_size=800,
            similarity_threshold=0.5
        )
        semantic_chunks = await semantic_chunker.chunk_text_semantic(text_blocks, metadata)
        
        # Calculate metrics
        return {
            "token_based": {
                "num_chunks": len(token_chunks),
                "avg_chunk_size": np.mean([c["metadata"]["token_count"] for c in token_chunks]),
                "std_chunk_size": np.std([c["metadata"]["token_count"] for c in token_chunks]),
                "method": "Fixed 600 tokens with 100 overlap"
            },
            "semantic": {
                "num_chunks": len(semantic_chunks),
                "avg_chunk_size": np.mean([c["metadata"]["token_count"] for c in semantic_chunks]),
                "std_chunk_size": np.std([c["metadata"]["token_count"] for c in semantic_chunks]),
                "method": "Similarity-based with threshold 0.5"
            },
            "summary": {
                "recommendation": "Semantic chunking creates more coherent topic-based chunks",
                "trade_off": "Semantic = better quality but slower (needs embedding)",
                "use_case": "Use semantic for complex docs, token-based for simple/fast ingestion"
            }
        }
