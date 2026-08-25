# Build from chaosglue root (sibling welvet / tide / webgpu):
#   docker build -f welvet/apps/aai/test51/Dockerfile -t test51 .
FROM golang:1.22-bookworm AS build

WORKDIR /src
COPY welvet /src/welvet
COPY tide /src/tide
COPY webgpu /src/webgpu

WORKDIR /src/welvet/apps/aai/test51
RUN go mod tidy && GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/test51 .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/test51 /app/test51
COPY welvet/apps/aai/test51/.env.example /app/.env.example
EXPOSE 5151 8080
ENV TEST51_ADDR=0.0.0.0:5151 \
    TIDE_ADDR=0.0.0.0:8080 \
    TEST51_CKPT=/app/test51_ckpt
ENTRYPOINT ["/app/test51"]
