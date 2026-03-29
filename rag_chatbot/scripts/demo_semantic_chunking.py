"""
Demo script: Compare Token-based vs Semantic Chunking

Usage:
    python scripts/demo_semantic_chunking.py

Shows:
- Side-by-side comparison of both strategies
- Chunk boundaries visualization
- Similarity scores between sentences
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.chunker import TextChunker
from backend.services.semantic_chunker import SemanticChunker, ChunkingComparison
from backend.services.embeddings import EmbeddingService
from backend.config import settings


# Sample document with clear topic transitions
SAMPLE_TEXT = """
The company PTO policy allows employees to take up to 15 paid days off per year. 
Employees must submit PTO requests at least two weeks in advance through the HR portal.
Unused PTO days can be carried over to the next year, but only up to 5 days maximum.
The PTO approval process typically takes 2-3 business days for manager review.

The company offers comprehensive health insurance to all full-time employees.
Health insurance includes medical, dental, and vision coverage at no cost to employees.
Employees can add family members to their health plan for an additional monthly fee.
Open enrollment for health benefits occurs annually in November.

Company parking is available in the underground garage on levels B1 through B3.
Employees must register their vehicle license plate with the security office.
Visitor parking is located on level B1 near the elevator entrance.
Electric vehicle charging stations are available on level B2.
"""


async def demo_chunking_strategies():
    """Main demo function."""
    
    print("=" * 80)
    print("SEMANTIC CHUNKING DEMO")
    print("=" * 80)
    print()
    
    # Initialize embedding service
    print("Initializing embedding service...")
    embedding_service = EmbeddingService()
    await embedding_service.initialize()
    print("✓ Embedding service ready\n")
    
    # Prepare text blocks
    text_blocks = [
        {
            "text": SAMPLE_TEXT.strip(),
            "type": "paragraph",
            "page_number": 1,
            "section": "Employee Handbook"
        }
    ]
    
    metadata = {
        "document_id": "demo-doc",
        "filename": "demo.txt",
        "source": "demo"
    }
    
    # =========================================================================
    # 1. TOKEN-BASED CHUNKING
    # =========================================================================
    print("🔵 TOKEN-BASED CHUNKING (Fixed 200 tokens)")
    print("-" * 80)
    
    token_chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    token_chunks = token_chunker.chunk_text(text_blocks, metadata)
    
    print(f"Number of chunks: {len(token_chunks)}\n")
    
    for i, chunk in enumerate(token_chunks, 1):
        content = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
        tokens = chunk["metadata"]["token_count"]
        print(f"Chunk {i} ({tokens} tokens):")
        print(f"  {content}")
        print()
    
    # =========================================================================
    # 2. SEMANTIC CHUNKING
    # =========================================================================
    print("=" * 80)
    print("🟢 SEMANTIC CHUNKING (Similarity threshold 0.5)")
    print("-" * 80)
    
    semantic_chunker = SemanticChunker(
        embedding_service=embedding_service,
        max_chunk_size=500,
        min_chunk_size=50,
        similarity_threshold=0.5,
        overlap_sentences=1
    )
    
    semantic_chunks = await semantic_chunker.chunk_text_semantic(text_blocks, metadata)
    
    print(f"Number of chunks: {len(semantic_chunks)}\n")
    
    for i, chunk in enumerate(semantic_chunks, 1):
        content = chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
        tokens = chunk["metadata"]["token_count"]
        print(f"Chunk {i} ({tokens} tokens):")
        print(f"  {content}")
        print()
    
    # =========================================================================
    # 3. DETAILED COMPARISON
    # =========================================================================
    print("=" * 80)
    print("📊 COMPARISON ANALYSIS")
    print("-" * 80)
    
    comparison = await ChunkingComparison.compare_strategies(
        text_blocks, 
        metadata, 
        embedding_service
    )
    
    print("\nToken-based:")
    for key, value in comparison["token_based"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nSemantic:")
    for key, value in comparison["semantic"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nSummary:")
    for key, value in comparison["summary"].items():
        print(f"  {key}: {value}")
    
    # =========================================================================
    # 4. SIMILARITY VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔍 SEMANTIC SIMILARITY BETWEEN SENTENCES")
    print("-" * 80)
    
    sentences = semantic_chunker._split_into_sentences(SAMPLE_TEXT.strip())
    embeddings = await semantic_chunker._embed_sentences(sentences)
    
    if embeddings is not None:
        print(f"\nTotal sentences: {len(sentences)}\n")
        
        for i in range(1, min(10, len(sentences))):
            similarity = semantic_chunker._calculate_similarity(
                embeddings[i-1], 
                embeddings[i]
            )
            
            # Visualize similarity
            bar_length = int(similarity * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            status = "✓ Same topic" if similarity >= 0.5 else "✗ Topic break"
            
            print(f"Sentence {i-1} → {i}:")
            print(f"  Similarity: {similarity:.3f} {bar} {status}")
            print(f"  S{i-1}: {sentences[i-1][:60]}...")
            print(f"  S{i}: {sentences[i][:60]}...")
            print()
    
    print("=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_chunking_strategies())
