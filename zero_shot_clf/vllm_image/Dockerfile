FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN pip3 install vllm transformers fastapi uvicorn[standard] \
    opentelemetry-instrumentation-fastapi opentelemetry-sdk opentelemetry-exporter-otlp

COPY llm_app.py /app/app.py
WORKDIR /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]