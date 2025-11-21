<!-- 3c31dbe8-b2ad-4687-a019-46310d510af9 37ab477c-a3b1-41cf-a241-6334762bc17a -->
# MIDI Manuals ColPali Plan

1. **Environment Hooks**  

- Keep the existing dependency set (already managed via `uv`) and note any runtime expectations.  
- Add `src/qdrant_init/settings.py` to load `.env`, instantiate the shared `QdrantClient`, and expose `Path` helpers for input folders plus utilities to create/clean `documents/midi_synthesizers/tmp/{pages,text}`.

2. **Manual preprocessing & indexing CLI**  

- Create `src/qdrant_init/manuals_ingest.py` with an `argparse` CLI (`uv run python -m qdrant_init.manuals_ingest --subset test ...`).  
- For each PDF: render its pages to RGB images in `tmp/pages` via `pypdfium2`, extract per-page text with `pymupdf` so we can build text payloads, dense/sparse embeddings, and human-readable snippets, then persist lightweight JSON manifests in `tmp/text`.  
- Use `colpali_engine` to generate multivectors for every page, storing original vectors (`colpali_original`, HNSW disabled) plus mean-pooled row/column vectors (`colpali_rows`, `colpali_cols`) for first-stage recall.  
- Chunk the extracted text (reuse `semantic-text-splitter`), embed chunks with `fastembed.TextEmbedding` (dense) and `fastembed.SparseTextEmbedding` (BM25), and align them with the same point IDs.  
- Ensure the script manages the Qdrant collection (dense + sparse + multivectors) and upserts batches of `PointStruct` objects carrying payload metadata (manual name, page, text, temp paths).

3. **Hybrid query + rerank utility**  

- Build `src/qdrant_init/manuals_query.py` CLI to run a single query against the chosen collection.  
- Compute dense & sparse embeddings for the question (FastEmbed) and ColPali multivectors for reranking; issue a `client.query_points` call with `prefetch` over dense + sparse (and optionally `colpali_rows/cols`) and final rerank via `using="colpali_original"`.  
- Pretty-print results with payload snippets (pulled from the per-page text) and write each run to `logs/manuals_queries/<timestamp>.json` for reproducibility.

4. **LangGraph agent with Brave search**  

- Implement `src/qdrant_init/manuals_agent.py` that wires a LangGraph state machine similar to `agentic-rag-langgraph.md`.  
- Register two tools: (a) `manuals_retriever_tool` wrapping the hybrid query module to return top-k contexts, and (b) a `BraveSearch` tool reading `BRAVE_API_KEY`.  
- Use `ChatOpenAI` (default `gpt-4o-mini`) for the agent node, configure routing (`ToolNode` + `route` fn) per the doc, and expose a CLI (`uv run python -m qdrant_init.manuals_agent --question ... --collection ...`) that streams the agent’s reasoning plus final grounded answer.

5. **Documentation & defaults**  

- Update `README.md` to explain the new workflow: temp directory layout, ingestion pipeline, querying utility, and agent usage (including `uv` commands and required environment variables).  
- Document expected order of operations (ingest `test` first, then `full`) and include notes on cleaning temp dirs and monitoring Qdrant collection sizes.

### To-dos

- [x] Add settings/env loader
- [ ] Build manuals_ingest CLI for ColPali ingestion
- [ ] Implement manuals_query hybrid rerank tool
- [ ] Create LangGraph manuals_agent with Brave tool
- [ ] Refresh README with workflow + commands