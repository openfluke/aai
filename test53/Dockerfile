# Build from parent of welvet/ (sibling tide/ + webgpu/):
#   docker build -f welvet/apps/aai/test53/Dockerfile -t test53 .
FROM golang:1.22-bookworm AS build

WORKDIR /src
COPY welvet /src/welvet
COPY tide /src/tide
COPY webgpu /src/webgpu

WORKDIR /src/welvet/apps/aai/test53
RUN go mod tidy && GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/test53 .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/test53 /app/test53
COPY welvet/apps/aai/test53/.env.example /app/.env.example
EXPOSE 8080
ENV TEST53_LAYERS=all \
    TEST53_MODES=all \
    TEST53_DTYPES=all \
    TEST53_WORKERS=4 \
    TEST53_DURATION=2s \
    TEST53_LR=0.05 \
    TEST53_HIDDEN=32 \
    TEST53_CKPT=/app/test53_ckpt \
    TEST53_RESUME=true \
    TEST53_TIDE_ADDR=0.0.0.0:8080
ENTRYPOINT ["/app/test53"]
