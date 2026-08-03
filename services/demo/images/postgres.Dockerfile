# eco-demo-postgres — stock Postgres with the schema baked into the init dir.
#
# The dev and Server A stacks mount db_init.sql from the checkout. A deployed
# demo has none, so it is copied in. This uses the *dev* init script, which
# seeds `test@test.com / test1234` — wanted for a demo, deliberately absent
# from services/server-a/db_init.sql.
#
# Runs once, on first start with an empty data dir (standard Postgres image
# behaviour). Re-deploying over an existing volume will not re-run it.
#
# Build (from the repo root):
#   docker buildx build -f services/demo/images/postgres.Dockerfile \
#     --build-context sql=services/dev \
#     -t eco-demo-postgres:local services/demo/images
FROM postgres:16-alpine

COPY --from=sql db_init.sql /docker-entrypoint-initdb.d/01_init.sql
