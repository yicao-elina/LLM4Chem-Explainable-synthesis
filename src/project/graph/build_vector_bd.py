import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- configuration ---
JSON_FILE_PATH = 'outputs/enriched_knowledge_graph.json'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHROMA_PATH = './checkpoints/enriched_knowledge_graph' 
COLLECTION_NAME = 'causal_relationships' 

def format_relationship_as_text(rel: dict) -> str:
    """Formats the single relationship JSON object as a text string for embedding."""
    cause = rel.get("cause_parameter", "N/A")
    effect = rel.get("effect_on_doping", "N/A")
    prop = rel.get("affected_property", "N/A")
    mechanism = rel.get("mechanism_quote")
    
    # Check if competing_processes exists and is a list
    competing_processes = rel.get("competing_processes")
    if isinstance(competing_processes, list):
        competing_processes = ", ".join(competing_processes) # Join list into a single string

    text = f"The cause of the relationship is {cause}. The effect of the relationship is {effect}. The affected property of the relationship is {prop}."
    if mechanism:
        text += f" The mechanism of the relationship is {mechanism}"
    if competing_processes:
        text += f" The competing processes of the relationship are {competing_processes}"
    return text

def preprocess_metadata(metadata: dict) -> dict:
    """
    Cleans metadata to ensure all values are ChromaDB-compatible types.
    Specifically converts lists to comma-separated strings and handles None values.
    """
    clean_meta = {}
    for key, value in metadata.items():
        if value is None:
            # ChromaDB doesn't accept None values, so skip or convert to empty string
            clean_meta[key] = ""
        elif isinstance(value, list):
            # Convert list to a single string
            clean_meta[key] = ", ".join(map(str, value)) if value else ""
        elif isinstance(value, (str, int, float, bool)):
            # Keep supported types as they are
            clean_meta[key] = value
        else:
            # For other unsupported types (like dicts), convert to string
            clean_meta[key] = str(value)
    return clean_meta

def main():
    print("🚀 start building vector database...")

    # 1. initialize model and database
    print(f"loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # To ensure a clean build, try to delete the old collection first
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 2. load and process data
    print(f"reading data source: {JSON_FILE_PATH}")
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    relationships = data.get("causal_relationships", data)
    
    documents = []
    metadatas = []
    ids = []

    for idx, rel in enumerate(relationships):
        doc_text = format_relationship_as_text(rel)
        
        # *** THIS IS THE FIX ***
        # Preprocess the metadata to handle lists and other types
        clean_metadata = preprocess_metadata(rel)
        
        documents.append(doc_text)
        metadatas.append(clean_metadata) # Use the cleaned metadata
        ids.append(str(idx))

    # 3. batch generate embeddings and store
    print(f"generating embeddings for {len(documents)} relationships...")
    embeddings = model.encode(documents, show_progress_bar=True)
    
    print(f"adding data to ChromaDB collection '{COLLECTION_NAME}'...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("\n✅ vector database built successfully!")
    print(f"processed {collection.count()} relationships.")
    print(f"database stored at: {CHROMA_PATH}")

if __name__ == '__main__':
    main()