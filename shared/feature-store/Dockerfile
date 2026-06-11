FROM python:3.11-slim

WORKDIR /app/feature-store
RUN pip install --no-cache-dir "feast[redis,gcp]==0.49.0" "protobuf<6"
COPY shared/feature-store/definitions.py shared/feature-store/feature_store.yaml ./
COPY shared/feature-store/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 6566
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
