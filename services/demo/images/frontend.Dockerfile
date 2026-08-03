# eco-demo-frontend — the unmodified Vite app, served by the dev server.
#
# Same as frontend/Dockerfile minus the entrypoint: that script re-runs
# `npm install` at boot for the bind-mounted dev stack, which a deployed demo
# neither needs nor can rely on. Baked node_modules is used as-is.
#
# Still the dev server, not a static build, because VITE_API_URL is read at
# process start — the deployed stack sets it to "" so the app calls /api
# same-origin through nginx. vite.config.js pins allowedHosts to
# `.adelaideuni.cloud`; nginx presents `Host: localhost` (always allowed) to
# this container, so the stack works behind any hostname.
#
# Build (from the repo root):
#   docker buildx build -f services/demo/images/frontend.Dockerfile \
#     --build-context app=frontend \
#     -t eco-demo-frontend:local services/demo/images
FROM node:20-alpine

WORKDIR /app

COPY --from=app package.json ./
RUN npm install

COPY --from=app . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
