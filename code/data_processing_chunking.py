import json
import re
from typing import List, Dict, Any
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Cleans and normalizes Wikipedia text for kid-friendly RAG.
    - Removes references, extra whitespace, and unwanted sections.
    - Optionally, more advanced cleaning can be added.
    """
    # Remove [edit] tags and reference numbers like [1], [2], etc.
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n{2,}', '\n', text)
    # Remove sections that are not kid-appropriate (e.g., 'See also', 'References', 'External links')
    text = re.split(r'\n(See also|References|External links|Further reading|Notes)\n', text, maxsplit=1)[0]
    # Strip leading/trailing whitespace
    return text.strip()

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping, semantically coherent chunks.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent.split()) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap: keep last N words
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words)] if overlap_words else []
            current_len = len(overlap_words)
        current_chunk.append(sent)
        current_len += len(sent.split())
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return [c.strip() for c in chunks if len(c.strip().split()) > 30]  # filter out tiny chunks

def process_wikipedia_json(in_path: str, out_path: str, chunk_size: int = 300, overlap: int = 50):
    """
    Loads Wikipedia JSON, cleans and chunks text, and outputs a new JSON with metadata for each chunk.
    """
    with open(in_path, encoding='utf-8') as f:
        articles = json.load(f)
    processed = []
    for article in articles:
        clean = clean_text(article['text'])
        chunks = chunk_text(clean, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            processed.append({
                "id": f"{article['id']}_chunk{i+1}",
                "title": article['title'],
                "chunk_index": i+1,
                "text": chunk,
                "source": article['source'],
                "language": article['language'],
                "retrieved_at": article['retrieved_at'],
                "wikidata_id": article.get('wikidata_id'),
                "fullurl": article.get('fullurl'),
                # Add more metadata as needed
            })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"Processed and chunked {len(processed)} text chunks saved to {out_path}")

if __name__ == "__main__":
    # Example usage for English Wikipedia
    process_wikipedia_json(
        in_path="../data/ethiopian_history_en.json",
        out_path="../vector_db/ethiopian_history_en_chunked.json",
        chunk_size=300,
        overlap=50
    )
    # Example usage for Amharic Wikipedia
    process_wikipedia_json(
        in_path="../data/ethiopian_history_am.json",
        out_path="../vector_db/ethiopian_history_am_chunked.json",
        chunk_size=300,
        overlap=50
    )
