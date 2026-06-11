from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # Try loading existing index; if missing and we have documents, build it.
    loaded = store.load()
    if not loaded:
        if docs:
            print("[INFO] Building Faiss index because none was found and documents are available.")
            store.build_from_documents(docs)
            store.load()
        else:
            print("[WARN] No Faiss index found and no documents in 'data/' to build one. Add files to 'data/' or create the index manually.")
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
