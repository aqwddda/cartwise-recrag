# CartWise Streamlit UI

This package contains the Streamlit demo for the CartWise recommendation API.

The UI is only an HTTP client. It calls FastAPI endpoints and does not import or
execute recommendation, retrieval, evidence, Qdrant, model, or LLM code.

## Start The Backend

Run commands from the repository root.

```powershell
.\.venv\Scripts\python.exe -m uvicorn cartwise.api.main:app --reload
```

Before using the UI, make sure the backend readiness endpoint is ready:

```text
GET http://127.0.0.1:8000/health/ready
```

If readiness returns `503`, check the initialization error for Qdrant, model,
index, data file, or LLM key issues before requesting recommendations.

## Start The Frontend

```powershell
.\.venv\Scripts\python.exe -m streamlit run cartwise/ui/app.py
```

The default API base URL is:

```text
http://127.0.0.1:8000
```

You can override it with the `CARTWISE_API_BASE_URL` environment variable or in
the Streamlit sidebar.

## UI Scope

The Streamlit page supports one recommendation turn:

- Enter a natural language shopping need.
- Choose `Top K` from 1 to 50.
- Optionally set `user_id` in the sidebar.
- View product cards with reason, potential cons, review evidence, retrieval
  details, diagnostics, latency, and developer details.

The UI does not implement login, persistence, Redis, multi-turn sessions, agents,
deployment features, query translation, retrieval, filtering, fusion, evidence
RAG, or LLM explanation logic.
