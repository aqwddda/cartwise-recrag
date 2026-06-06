# CartWise API Boundary

This package contains HTTP-facing FastAPI code only.

Planned responsibilities:

- Keep `/health` only as an optional compatibility liveness route.
- Use `/health/live` for process liveness.
- Use `/health/ready` for readiness of Qdrant, Dense, BM25, Popularity,
  LightGCN, LLM adapters, `RecommendationService`, `EvidenceService`, and the
  top-level application service.
- Initialize heavy resources once in FastAPI lifespan during the API stage.
- Route handlers should call the already constructed application service.
- Convert application-service results into explicit response schemas instead of
  returning internal service objects directly.

Dependency direction:

- API code may depend on `cartwise.application`.
- API code must not assemble retrieval channels, load models, rebuild indexes,
  or create Qdrant/LLM clients per request.
- API schemas must not expose smoke-only modes, debug channel switches, or
  internal fusion/retrieval configuration.

TODO for the API stage:

- Add request/response schemas.
- Add a lifespan composition root for real resources.
- Add fake application service injection for tests.
- Implement `POST /api/v1/recommend`.
