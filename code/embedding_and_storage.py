import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Choose a multilingual embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKED_DATA_PATHS = [
    "../vector_db/ethiopian_history_en_chunked.json",
    "../vector_db/ethiopian_history_am_chunked.json"
]
VECTOR_DB_DIR = "../vector_db"
COLLECTION_NAME = "ethiopian_history"


def embed_and_store_chunks():
    # Load the embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Set up ChromaDB persistent client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR, settings=Settings(allow_reset=True))
    # Create or get collection
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
    for path in CHUNKED_DATA_PATHS:
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {path}")
        for chunk in chunks:
            # Embed the chunk text
            embedding = model.encode(chunk["text"]).tolist()
            # Store in ChromaDB
            collection.add(
                ids=[chunk["id"]],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[{
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"],
                    "source": chunk["source"],
                    "language": chunk["language"],
                    "retrieved_at": chunk["retrieved_at"],
                    "wikidata_id": chunk.get("wikidata_id"),
                    "fullurl": chunk.get("fullurl"),
                }]
            )
        print(f"Embedded and stored {len(chunks)} chunks from {path}")
    print(f"Final document count in collection: {collection.count()}")

if __name__ == "__main__":
    embed_and_store_chunks()
