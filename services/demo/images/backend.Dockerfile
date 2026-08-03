# eco-demo-backend — the unmodified Express backend, plus the sample tier the
# demo serves from disk.
#
# Two differences from backend/Dockerfile, both because a deployed demo has no
# repo checkout to bind-mount:
#
#   1. The `layers` fixture tree is copied to /mock/layers and AI_LAYERS_ROOT
#      points at it, so GET /api/samples works with no `dvc pull` and no
#      acoustic_ai/ on the host.
#   2. No entrypoint. backend/entrypoint.sh wipes node_modules and re-runs
#      `npm install` on every start, which only makes sense with a bind-mounted
#      source tree; here it would just make the container need a registry it
#      may not be able to reach. The baked node_modules is used as-is.
#
# Backend source itself is untouched — this is still `npm run dev`.
#
# Build (from the repo root):
#   docker buildx build -f services/demo/images/backend.Dockerfile \
#     --build-context app=backend \
#     --build-context fixtures=services/dev/ai-mock/fixtures/layers \
#     -t eco-demo-backend:local services/demo/images
FROM node:20-alpine

WORKDIR /app

RUN apk add --no-cache curl

COPY --from=app package.json ./
RUN npm install

COPY --from=app . .
COPY --from=fixtures . /mock/layers

# Must end in `layers` or backend/src/samples.js appends a second one.
ENV AI_LAYERS_ROOT=/mock/layers
# NOT `production`: index.js sets the session cookie `secure` under
# NODE_ENV=production, which silently breaks login on a plain-HTTP demo host.
ENV NODE_ENV=development

EXPOSE 4000

CMD ["npm", "run", "dev"]
