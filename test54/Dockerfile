# Runtime-only image. Binary is compiled by ./run-docker.sh --build (not here).
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY test54 /app/test54
COPY .env.example /app/.env.example
EXPOSE 8080
ENV TEST54_LAYERS=mamba \
    TEST54_MODES=all \
    TEST54_DTYPES=all \
    TEST54_LRS=0.05 \
    TEST54_WORKERS=4 \
    TEST54_DURATION=15s \
    TEST54_LR=0.05 \
    TEST54_HIDDEN=32 \
    TEST54_DEPTH=4 \
    TEST54_CKPT=/app/test54_ckpt \
    TEST54_CKPT_FLAT=true \
    TEST54_RESUME=true \
    TEST54_TIDE_ADDR=0.0.0.0:8080
ENTRYPOINT ["/app/test54"]
