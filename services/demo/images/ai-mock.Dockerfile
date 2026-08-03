# eco-demo-ai-mock — the fake AI service with its fixtures baked in.
#
# The dev stack (services/dev/docker-compose.yml) bind-mounts the fixture tree
# and registry.yaml from the checkout. A deployed demo has no checkout, so this
# image carries both. Same paths, so app/settings.py needs no change.
#
# Build (from the repo root):
#   docker buildx build -f services/demo/images/ai-mock.Dockerfile \
#     --build-context app=services/dev/ai-mock \
#     --build-context registry=acoustic_ai \
#     -t eco-demo-ai-mock:local services/demo/images
FROM python:3.11-slim

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=app requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=app app ./app
COPY --from=app fixtures /fixtures
# The registry is what GET /layers is built from. Baked, not mounted — but it
# is still the real acoustic_ai/registry.yaml, so the attempt list a deployed
# demo shows matches the branch it was built from.
COPY --from=registry registry.yaml /registry.yaml

ENV MOCK_FIXTURES_ROOT=/fixtures \
    MOCK_REGISTRY_PATH=/registry.yaml \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
