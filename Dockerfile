FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY params.yaml .

# Copy model artifacts (built by the CI pipeline)
COPY models/ ./models/
COPY reports/metrics.json ./reports/metrics.json

ENV MODEL_PATH=models/model.pkl
ENV PREPROCESSOR_PATH=models/preprocessor.pkl
ENV METRICS_PATH=reports/metrics.json
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
