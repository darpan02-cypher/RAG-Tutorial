# RAG-Tutorials

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline implemented in Python using LangChain components, Sentence-Transformers embeddings, and a FAISS vector store. The pipeline is split into two main stages: the Data Ingestion Pipeline and the Retrieval Pipeline. The code is in the `src/` folder and notebooks illustrate concepts and examples.

## Architecture Overview

- **Data Ingestion Pipeline**: Load raw files (PDF, TXT, CSV, Excel, DOCX, JSON) → convert to LangChain `Document` objects (with `page_content` and `metadata`) → chunk large documents → compute embeddings per chunk → persist vectors + metadata into a vector store.
- **Retrieval Pipeline**: Convert user query to an embedding (same model as ingestion) → perform similarity search in the vector store (FAISS) → retrieve top-k relevant chunks → pass retrieved context to an LLM to summarize or answer.

## Key components and where they live

- File: [src/data_loader.py](src/data_loader.py) — loads PDFs, TXT, CSV, XLSX, DOCX, JSON and converts them to LangChain `Document` objects.
- File: [src/embedding.py](src/embedding.py) — chunking (via `RecursiveCharacterTextSplitter`) and embedding generation (default model: `all-MiniLM-L6-v2` using `sentence-transformers`). Chunk config: `chunk_size=1000`, `chunk_overlap=200` by default.
- File: [src/vectorstore.py](src/vectorstore.py) — FAISS-backed vector store implementation. Uses `faiss.IndexFlatL2` and persists index to `faiss_store/` along with `metadata.pkl`.
- File: [src/search.py](src/search.py) — RAG search wrapper that loads/builds the vectorstore, queries FAISS, combines top-k retrieved chunks into context, and calls a ChatGroq LLM (`gemma2-9b-it` in code) to summarize/respond.
- File: [app.py](app.py) — example entrypoint demonstrating load/build and a sample query.

## Data Ingestion Pipeline (detailed)

1. Data Loading and Parsing
	- Implemented in [src/data_loader.py](src/data_loader.py).
	- Supported formats: PDF, TXT, CSV, Excel (.xlsx), DOCX, JSON.
	- Each loader returns LangChain `Document` objects with `page_content` and optional `metadata`.

2. Chunking
	- Implemented in [src/embedding.py](src/embedding.py) with `RecursiveCharacterTextSplitter`.
	- Default `chunk_size=1000` and `chunk_overlap=200` to keep chunk lengths within typical LLM context limits.

3. Embedding Generation
	- Using `sentence-transformers` (`SentenceTransformer`) with default model `all-MiniLM-L6-v2`.
	- Implemented in `EmbeddingPipeline.embed_chunks()` which encodes chunk texts into numeric vectors.

4. Vector Store Persistence
	- Implemented in [src/vectorstore.py](src/vectorstore.py) using FAISS (`faiss-cpu`).
	- The index file (`faiss.index`) and metadata (`metadata.pkl`) are stored inside the configured `persist_dir` (default `faiss_store/`).

## Retrieval Pipeline (detailed)

1. Query Embedding
	- Queries are encoded with the same `SentenceTransformer` model as used for ingestion (ensures embedding space alignment).

2. Similarity Search
	- FAISS `IndexFlatL2` is used for nearest-neighbor search (L2 distance). The vectorstore returns the `top_k` results and corresponding metadata.

3. Context Retrieval and LLM Response
	- Retrieved chunk texts are concatenated to form a context.
	- The pipeline calls an LLM to summarize or answer based on the retrieved context. The example uses `langchain_groq.ChatGroq` with `gemma2-9b-it` in [src/search.py](src/search.py).

## How to run (quickstart)

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Prepare your documents
	- Place your data inside a `data/` directory at the repo root (the loader searches `data/**`).

3. Build or run the app
	- The `app.py` entrypoint will load the vectorstore if present, otherwise it will build it from `data/`.

```bash
python app.py
```

Notes:
- If the `faiss.index` and `metadata.pkl` files are missing in `faiss_store/`, `RAGSearch` will build the vectorstore automatically.
- To force a rebuild manually, remove the files in `faiss_store/` and re-run `python app.py`.

## Configuration & customization

- Change embedding model: edit the `model_name` parameter in `EmbeddingPipeline` (file: [src/embedding.py](src/embedding.py)) or pass a different model name when instantiating `EmbeddingPipeline` / `FaissVectorStore`.
- Adjust chunking: modify `chunk_size` and `chunk_overlap` in `EmbeddingPipeline` or when constructing `FaissVectorStore`.
- Change LLM: update `llm_model` and provide a valid Groq API key (see `src/search.py`; currently `groq_api_key` is a placeholder). Use a `.env` and `python-dotenv` to load keys.

## Requirements

See [requirements.txt](requirements.txt). Important packages used:

- `sentence-transformers` — embedding model (default: `all-MiniLM-L6-v2`)
- `faiss-cpu` — vector similarity search and persistence
- `langchain`, `langchain-community` — document loaders and helpers
- `langchain-groq` — Groq LLM integration used in `src/search.py`

## File map

- [src/data_loader.py](src/data_loader.py) — file loaders and conversion to LangChain `Document` objects
- [src/embedding.py](src/embedding.py) — chunking + embedding generation
- [src/vectorstore.py](src/vectorstore.py) — FAISS-backed vector store (persist/load/query)
- [src/search.py](src/search.py) — RAG wrapper: retrieval + LLM summarization
- [app.py](app.py) — example script showing build/load and a sample query
- `books.jsonl`, notebooks/ — example data and explanatory notebooks

## Example flow (commands)

```bash
# install
python3 -m pip install -r requirements.txt

# add files to ./data (PDFs, TXT, CSV, etc.)

# run app (will build vectorstore if missing)
python app.py

# after build: use RAGSearch in your own script or adapt app.py
```

## Troubleshooting

- If embeddings fail: ensure `sentence-transformers` is installed and the chosen model is available.
- If FAISS operations fail on macOS: install `faiss-cpu` from pip or use a conda environment if needed.
- If LLM step fails: set a valid Groq API key and confirm network access.

---

If you'd like, I can also add small example scripts to show how to (a) build the store from a specific folder, (b) run ad-hoc queries, or (c) switch to a different vector DB (Chroma/Weaviate). Which would you prefer next?