FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN pip3 install transformers fastapi uvicorn[standard] \
    opentelemetry-instrumentation-fastapi opentelemetry-sdk opentelemetry-exporter-otlp

RUN pip3 install --no-cache-dir \
        "git+https://github.com/deepvk/geracl.git@main"

RUN pip3 install datasets pytorch_lightning torchmetrics loguru PyYAML scikit_learn

COPY geracl_app.py /app/app.py
WORKDIR /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]