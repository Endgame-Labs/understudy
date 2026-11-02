# Runtime image for `understudy serve`. The listener has no authentication: publish the port only to trusted
# networks, or put an authenticating proxy in front of it.
#
# The binary is not compiled here. The release workflow extracts the release archives into
# dist/amd64/understudy and dist/arm64/understudy before building.
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin understudy \
 && mkdir /data && chown understudy /data

ARG TARGETARCH
COPY --chmod=0755 dist/${TARGETARCH}/understudy /usr/local/bin/understudy
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Snapshots, the capture log, and the encoder download (about 1.5 GB) live on the volume.
ENV HF_HOME=/data/hf
USER understudy
WORKDIR /data
VOLUME /data
EXPOSE 8787

LABEL org.opencontainers.image.source="https://github.com/endgame-labs/understudy" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
