from fastapi import FastAPI
from geracl import GeraclHF, ZeroShotClassificationPipeline
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from transformers import AutoTokenizer

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)

model = GeraclHF.from_pretrained("deepvk/GeRaCl-USER2-base").eval().to("cuda")
tokenizer = AutoTokenizer.from_pretrained("deepvk/GeRaCl-USER2-base")

pipe = ZeroShotClassificationPipeline(model, tokenizer, device="cuda")


@app.post("/classify")
def classify(payload: dict):
    texts = payload["texts"]
    labels = payload["labels"]
    batch_size = 1
    if "batch_size" in payload:
        batch_size = payload["batch_size"]

    result = pipe(texts, labels=labels, batch_size=batch_size)

    text_results = []
    print(len(texts))
    print(result)
    if isinstance(labels[0], str):
        for i in range(len(texts)):
            print(text_results.append(labels[result[i]]))
    else:
        for i in range(len(texts)):
            print(text_results.append(labels[i][result[i]]))

    return {
        "result": text_results,
    }
