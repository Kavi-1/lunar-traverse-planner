FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/index.html web/vite.config.js ./
COPY web/src ./src
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.11-slim-bookworm AS dependencies
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    LUNAR_DATA_DIR=/app/data/runtime \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
# Rasterio's bundled GDAL library dynamically links to system libexpat.so.1.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=dependencies /app/.venv /app/.venv
COPY core ./core
COPY api ./api
COPY --from=frontend /web/dist ./web/dist
# Explicit files make an absent bundle a build failure. See README preparation.
COPY data/runtime/site04_dem_window.tif \
     data/runtime/site04_mean_local_shadow.tif \
     data/runtime/site04_illumination_report.json \
     data/runtime/manifest.json ./data/runtime/
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8"]
