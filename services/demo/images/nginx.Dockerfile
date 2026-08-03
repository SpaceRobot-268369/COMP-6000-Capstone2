# eco-demo-nginx — the demo reverse proxy with its config baked in, so the
# deployed stack is images + .env and nothing else.
#
# Build (from the repo root):
#   docker buildx build -f services/demo/images/nginx.Dockerfile \
#     --build-context conf=services/demo/nginx \
#     -t eco-demo-nginx:local services/demo/images
FROM nginx:1.27-alpine

COPY --from=conf default.conf /etc/nginx/conf.d/default.conf
