FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir google-cloud-bigquery confluent-kafka
COPY pipeline-monitoring/producer-module/producer.py ./

CMD ["python", "-u", "producer.py"]
