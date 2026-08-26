# Runtime-only image. Binary is compiled by ./run-docker.sh --build (not here).
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY test53 /app/test53
COPY .env.example /app/.env.example
EXPOSE 8080
ENV TEST53_LAYERS=dense \
    TEST53_MODES=all \
    TEST53_DTYPES=all \
    TEST53_LRS=funny \
    TEST53_WORKERS=4 \
    TEST53_DURATION=2s \
    TEST53_LR=0.05 \
    TEST53_HIDDEN=32 \
    TEST53_CKPT=/app/test53_ckpt \
    TEST53_CKPT_FLAT=true \
    TEST53_RESUME=true \
    TEST53_TIDE_ADDR=0.0.0.0:8080
ENTRYPOINT ["/app/test53"]
