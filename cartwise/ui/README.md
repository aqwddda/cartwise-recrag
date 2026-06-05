# CartWise UI Boundary

This package is reserved for the future Streamlit demo.

Planned responsibilities:

- Collect user input.
- Call the FastAPI backend over HTTP.
- Render recommendations, evidence, diagnostics, and latency.

Dependency direction:

- UI code may call FastAPI HTTP endpoints.
- UI code must not import `cartwise.retrieval`, `cartwise.recommendation`,
  `cartwise.evidence`, Qdrant clients, model objects, or LLM clients directly.

TODO for the Streamlit stage:

- Add `cartwise/ui/app.py`.
- Read the backend base URL from configuration or environment.
- Keep all recommendation logic in the backend.
