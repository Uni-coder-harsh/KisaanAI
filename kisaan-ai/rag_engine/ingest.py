"""
KisaanAI — PDF Ingestion Pipeline
Loads agriculture PDFs → chunks text → embeds with sentence-transformers
→ upserts to Pinecone index for RAG retrieval.

Usage:
  python rag_engine/ingest.py --pdf path/to/file.pdf
  python rag_engine/ingest.py --dir data_pipeline/datasets/docs/
  python rag_engine/ingest.py --check   # just verify index status
"""

import os
import sys
import argparse
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
INDEX_NAME   = os.getenv("PINECONE_INDEX", "kisaan-ai")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, free
CHUNK_SIZE   = 500    # characters per chunk
CHUNK_OVERLAP = 80
BATCH_SIZE   = 100    # upsert batch size


# ── Text chunking ────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> List[Dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    chunk_id = 0
    text = text.strip()

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > CHUNK_SIZE // 2:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunk = chunk.strip()
        if len(chunk) > 50:  # skip tiny chunks
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
            chunks.append({
                "id":       f"{Path(source).stem}_{chunk_id}_{chunk_hash}",
                "text":     chunk,
                "source":   Path(source).name,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

        start = end - CHUNK_OVERLAP

    logger.info(f"  → {len(chunks)} chunks from {Path(source).name}")
    return chunks


# ── PDF extraction ───────────────────────────────────────────────
def extract_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i+1}]\n{text}")
        full = "\n\n".join(pages)
        logger.info(f"Extracted {len(full)} chars from {len(reader.pages)} pages")
        return full
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)


# ── Embedding ────────────────────────────────────────────────────
def get_embedder():
    """Load sentence-transformers embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedder: {EMBED_MODEL}")
        model = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedder loaded ✓")
        return model
    except ImportError:
        logger.error("sentence-transformers not installed.")
        sys.exit(1)


def embed_chunks(embedder, chunks: List[Dict]) -> List[Dict]:
    """Add embeddings to chunks."""
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec.tolist()
    return chunks


# ── Pinecone ─────────────────────────────────────────────────────
def get_pinecone_index():
    """Connect to or create Pinecone index."""
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        logger.error("pinecone not installed. Run: pip install pinecone")
        sys.exit(1)

    if not PINECONE_KEY:
        logger.error("PINECONE_API_KEY not set in .env")
        sys.exit(1)

    pc = Pinecone(api_key=PINECONE_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    logger.info(f"Existing indexes: {existing}")

    if INDEX_NAME not in existing:
        logger.info(f"Creating index '{INDEX_NAME}' (dim=384, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            logger.info("Waiting for index to be ready...")
            time.sleep(2)
        logger.info(f"Index '{INDEX_NAME}' created ✓")
    else:
        logger.info(f"Index '{INDEX_NAME}' already exists ✓")

    return pc.Index(INDEX_NAME)


def upsert_to_pinecone(index, chunks: List[Dict]):
    """Upsert embedded chunks to Pinecone in batches."""
    vectors = [
        {
            "id":     chunk["id"],
            "values": chunk["embedding"],
            "metadata": {
                "text":     chunk["text"],
                "source":   chunk["source"],
                "chunk_id": chunk["chunk_id"],
            },
        }
        for chunk in chunks
    ]

    total = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        total += len(batch)
        logger.info(f"  Upserted {total}/{len(vectors)} vectors")

    logger.info(f"✅ Upserted {total} vectors to Pinecone index '{INDEX_NAME}'")


def check_index():
    """Print current index stats."""
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    print("\n── Pinecone Index Stats ──────────────────")
    print(f"  Index name     : {INDEX_NAME}")
    print(f"  Total vectors  : {stats.total_vector_count}")
    print(f"  Dimension      : {stats.dimension}")
    print(f"  Namespaces     : {dict(stats.namespaces)}")
    print("──────────────────────────────────────────\n")


# ── Main ─────────────────────────────────────────────────────────
def ingest_file(pdf_path: str, embedder, index):
    logger.info(f"\n📄 Ingesting: {pdf_path}")
    text   = extract_pdf(pdf_path)
    chunks = chunk_text(text, pdf_path)
    chunks = embed_chunks(embedder, chunks)
    upsert_to_pinecone(index, chunks)


def main():
    parser = argparse.ArgumentParser(description="KisaanAI PDF Ingestion")
    parser.add_argument("--pdf",   help="Path to a single PDF file")
    parser.add_argument("--dir",   help="Directory of PDFs to ingest")
    parser.add_argument("--check", action="store_true", help="Check index stats")
    args = parser.parse_args()

    if args.check:
        check_index()
        return

    if not args.pdf and not args.dir:
        parser.print_help()
        sys.exit(1)

    index    = get_pinecone_index()
    embedder = get_embedder()

    if args.pdf:
        if not Path(args.pdf).exists():
            logger.error(f"File not found: {args.pdf}")
            sys.exit(1)
        ingest_file(args.pdf, embedder, index)

    elif args.dir:
        pdfs = list(Path(args.dir).glob("**/*.pdf"))
        if not pdfs:
            logger.error(f"No PDFs found in {args.dir}")
            sys.exit(1)
        logger.info(f"Found {len(pdfs)} PDFs in {args.dir}")
        for pdf in pdfs:
            ingest_file(str(pdf), embedder, index)

    check_index()
    logger.info("\n🌾 Ingestion complete! KisaanGPT is now smarter.")


if __name__ == "__main__":
    main()
