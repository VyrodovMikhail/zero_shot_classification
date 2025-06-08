import os

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from vllm import LLM, SamplingParams

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)
model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")

# Инициализируем движок vLLM:
llm = LLM(
    model="/models/Qwen3-4B",
    # model=model_id,
    dtype="bfloat16",  # в зависимости от используемой модели
    trust_remote_code=True,
)


def build_prompt(text: str, labels) -> str:
    labels_str = ", ".join(labels)
    return (
        "Присвой тексту одну из категорий ниже.\n"
        f"Возможные категории: {labels_str}.\n\n"
        f"Текст:\n{text}\n\n"
        "Ответ (только категория):"
    )


sampler = SamplingParams(
    temperature=0.0,  # делаем вывод максимально детерминированным
    top_p=1.0,
    max_tokens=30,
    stop=["\n"],  # модель останавливается после первой строки ответа
)


@app.post("/classify")
def classify(payload: dict):
    texts = payload["texts"]
    labels = payload["labels"]  # список меток для классификации
    batch_size = 1
    if "batch_size" in payload:
        batch_size = payload["batch_size"]

    # Формируем подсказку (prompt) для LLM, чтобы он выбрал метку
    if isinstance(labels[0], str):
        prompts = [build_prompt(t, labels) for t in texts]
    else:
        prompts = [build_prompt(text, categories) for text, categories in zip(texts, labels)]

    # --- batched generation -------------------------------------------------
    categories: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params=sampler)
        categories.extend(out.outputs[0].text.strip().lower() for out in outputs)

    # out = llm.generate([prompt], sampling_params=sampler)[0]
    return {
        "result": categories,
    }
