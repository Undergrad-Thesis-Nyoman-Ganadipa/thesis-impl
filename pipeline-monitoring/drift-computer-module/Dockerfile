FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir confluent-kafka requests numpy scipy prometheus-client
COPY pipeline-monitoring/drift-computer-module/*.py ./

EXPOSE 8000
CMD ["python", "-u", "detector.py"]
