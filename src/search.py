import os
from dotenv import load_dotenv
from .vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from .data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        try:
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
            print(f"[INFO] Groq LLM initialized: {llm_model}")
        except Exception as e:
            print(f"[WARN] Failed to initialize Groq LLM: {e}")
            print("[WARN] Continuing without LLM; search will return raw context.")
            self.llm = None

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        if self.llm is None:
            # Fallback: return the retrieved context as-is for now
            return f"Retrieved context for query '{query}':\n\n{context}"
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        try:
            response = self.llm.invoke([prompt])
            return response.content
        except Exception as e:
            print(f"[WARN] LLM invocation failed: {e}")
            return f"Retrieved context for query '{query}' (LLM unavailable):\n\n{context}"

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
