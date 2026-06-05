# CartWise API Boundary

This package contains HTTP-facing FastAPI code only.

Planned responsibilities:

- Keep the current `/health` route compatible while future API work separates
  liveness and readiness.
- Use `/health/live` for process liveness.
- Use `/health/ready` for readiness of Qdrant, Dense, BM25, Popularity,
  LightGCN, LLM adapters, `RecommendationService`, `EvidenceService`, and the
  top-level application service.
- Initialize heavy resources once in FastAPI lifespan during the API stage.
- Route handlers should call the already constructed application service.

Dependency direction:

- API code may depend on `cartwise.application`.
- API code must not assemble retrieval channels, load models, rebuild indexes,
  or create Qdrant/LLM clients per request.

TODO for the API stage:

- Add request/response schemas.
- Add a lifespan composition root for real resources.
- Add fake application service injection for tests.
- Implement `POST /api/v1/recommend`.
