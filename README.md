## Infinity Knowledge Mesh

Infinity Knowledge Mesh crawls any public web page, extracts named entities, and builds a lightweight knowledge graph that can be explored via the command line or a Streamlit UI.

### Features
- Robust crawler with URL validation, deduplicated outbound links, and request logging.
- HTTP retries, content-type/size safeguards, and optional same-domain link filtering.
- spaCy-powered entity extraction with automatic model download and heuristic fallback.
- Directed knowledge graph that links source pages to entities and discovered hyperlinks.
- Streamlit UI with presets, top-entity tables, customizable graph layouts, and an interactive PyVis network view.

### Getting Started
1) **Install dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate  # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2) **Run the Streamlit UI**
```bash
streamlit run app.py
```
Open the browser link Streamlit prints. Paste any URL, choose whether to include outbound links or limit to the same domain, and explore entities plus the PyVis graph.

3) **Command-line workflow**
```bash
python main.py --url https://en.wikipedia.org/wiki/OpenAI --max-entities 75
```
Flags:
- `--url`: target page to crawl.
- `--max-entities`: limit entity count (default 50).
- `--skip-links`: skip adding outgoing hyperlinks to the graph.
- `--same-domain-only`: keep only outbound links on the page's domain.
- `--top-entities`: number of entities to highlight in the summary.
- `--top-domains`: number of outbound domains to list.

### Deploying
- **Docker (local or server)**
  ```bash
  docker build -t infinity-knowledge-mesh .
  docker run -p 8501:8501 infinity-knowledge-mesh
  ```
  Then browse to http://localhost:8501.

- **Streamlit Cloud**
  1. Push this repo to GitHub.
  2. In Streamlit Community Cloud, create a new app pointing to `app.py`.
  3. Add `en_core_web_sm` as a post-install step: `python -m spacy download en_core_web_sm`.

### Project Layout
- `crawler.py` - fetches page text and normalized outbound links.
- `entities.py` - extracts named entities using spaCy (with heuristic fallback).
- `graph_builder.py` - stores nodes and relations in a NetworkX directed graph.
- `app.py` - Streamlit UI for building/visualizing the mesh.
- `utils.py` - helpers for normalizing and validating URLs.
- `main.py` - CLI entry point for batch runs.

### Notes
- The Streamlit UI requires a modern browser.
- spaCy's language model download happens automatically if missing, but running the download command manually upfront avoids extra startup time.
