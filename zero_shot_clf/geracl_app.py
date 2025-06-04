from fastapi import FastAPI
from geracl import GeraclHF, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

app = FastAPI()

model = GeraclHF.from_pretrained("deepvk/GeRaCl-USER2-base").eval().to("cuda")
tokenizer = AutoTokenizer.from_pretrained("deepvk/GeRaCl-USER2-base")

pipe = ZeroShotClassificationPipeline(model, tokenizer, device="cuda")


@app.post("/classify")
def classify(payload: dict):
    texts = payload["texts"]
    labels = payload["labels"]
    batch_size = None
    if "batch_size" in payload:
        batch_size = payload["batch_size"]

    result = pipe(texts, candidate_labels=labels, batch_size=batch_size)

    text_results = []
    for i in range(len(labels)):
        print(text_results.append(labels[i][result[i]]))

    return {
        "result": text_results,
    }
